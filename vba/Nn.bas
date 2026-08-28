Attribute VB_Name = "Nn"
Option Explicit

' Die nichtlinearen Bausteine: LayerNorm, Softmax, GELU, kausale Maske.
' Formen wie in Mat.bas: 1-basiert, (Zeile, Spalte), Zeile = Zeitschritt.
'
' Jede Formel hier muss bitgenau dem entsprechen, was das Referenzmodell
' rechnet -- sonst weicht der Stapel weiter oben ab und man sucht an der
' falschen Stelle. Die drei Fallen sind unten einzeln kommentiert.

Private Const LN_EPS As Double = 0.00001            ' 1e-5, wie im Referenzmodell
Private Const GELU_C As Double = 0.044715
Private Const EXP_FLOOR As Double = -700#           ' darunter unterlaeuft Exp

' ---------------------------------------------------------------------------
' Tangens hyperbolicus. VBA bringt keinen mit.
'
' Ueber tanh(z) = (e^2z - 1) / (e^2z + 1) waere e^2z fuer grosse z ein
' Ueberlauf; ab |z| = 20 ist tanh auf 17 Nachkommastellen bereits +-1, deshalb
' die Abkuerzung. Fuer negative z wird die ungerade Symmetrie benutzt, damit
' e^2z klein bleibt statt gross zu werden.
Public Function Tanh(ByVal z As Double) As Double
    Dim e As Double
    If z > 20# Then
        Tanh = 1#
    ElseIf z < -20# Then
        Tanh = -1#
    ElseIf z >= 0# Then
        e = Exp(-2# * z)
        Tanh = (1# - e) / (1# + e)
    Else
        e = Exp(2# * z)
        Tanh = (e - 1#) / (e + 1#)
    End If
End Function

' ---------------------------------------------------------------------------
' GELU in der tanh-Naeherung:
'   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
'
' NICHT die erf-Variante. Die beiden unterscheiden sich in der dritten
' Nachkommastelle -- genug, um den Vergleich in der letzten Schicht zu
' zerlegen, aber wenig genug, um beim Draufschauen plausibel auszusehen.
' Eingabe:  x (T, C)   Ausgabe: (T, C)
Public Function Gelu(x() As Double) As Double()
    Dim i As Long, j As Long
    Dim v As Double, out() As Double
    Dim k As Double

    k = Sqr(2# / 3.14159265358979)
    ReDim out(1 To UBound(x, 1), 1 To UBound(x, 2))
    For j = 1 To UBound(x, 2)
        For i = 1 To UBound(x, 1)
            v = x(i, j)
            out(i, j) = 0.5 * v * (1# + Tanh(k * (v + GELU_C * v * v * v)))
        Next i
    Next j
    Gelu = out
End Function

' ---------------------------------------------------------------------------
' LayerNorm ueber die letzte Achse.
'
' Die Varianz wird OHNE Bessel-Korrektur gebildet, also durch C geteilt und
' nicht durch C-1. Mit der Korrektur waere das Ergebnis bei C = 128 um etwa
' 0.4 Prozent daneben -- klein genug, um unentdeckt zu bleiben, gross genug,
' um jede Toleranz von 1e-4 zu reissen.
' Eingabe:  x (T, C), w (1, C), b (1, C)   Ausgabe: (T, C)
Public Function LayerNorm(x() As Double, w() As Double, b() As Double) As Double()
    Dim t As Long, j As Long, nt As Long, nc As Long
    Dim mean As Double, var As Double, d As Double, inv As Double
    Dim out() As Double

    nt = UBound(x, 1)
    nc = UBound(x, 2)
    If UBound(w, 2) <> nc Or UBound(b, 2) <> nc Then
        Err.Raise vbObjectError + 520, "Nn.LayerNorm", _
                  "Gewicht/Bias passen nicht zur Breite " & nc
    End If

    ReDim out(1 To nt, 1 To nc)
    For t = 1 To nt
        mean = 0#
        For j = 1 To nc
            mean = mean + x(t, j)
        Next j
        mean = mean / nc

        var = 0#
        For j = 1 To nc
            d = x(t, j) - mean
            var = var + d * d
        Next j
        var = var / nc                       ' ohne Bessel-Korrektur

        inv = 1# / Sqr(var + LN_EPS)
        For j = 1 To nc
            out(t, j) = (x(t, j) - mean) * inv * w(1, j) + b(1, j)
        Next j
    Next t

    LayerNorm = out
End Function

' ---------------------------------------------------------------------------
' Setzt in einer Score-Matrix alles oberhalb der Diagonale auf MASK_VALUE:
' Position t darf nur auf s <= t schauen.
'
' In VBA-Zaehlung entspricht das (t, s) mit s > t, weil beide Achsen um eins
' verschoben sind und die Verschiebung sich damit aufhebt.
' Eingabe/Ausgabe: (T, T), Zeile = Abfrageposition, Spalte = Schluesselposition
Public Function CausalMask(scores() As Double) As Double()
    Dim t As Long, s As Long, n As Long
    Dim out() As Double

    n = UBound(scores, 1)
    If UBound(scores, 2) <> n Then
        Err.Raise vbObjectError + 521, "Nn.CausalMask", "Matrix ist nicht quadratisch."
    End If

    ReDim out(1 To n, 1 To n)
    For s = 1 To n
        For t = 1 To n
            If s > t Then
                out(t, s) = MASK_VALUE
            Else
                out(t, s) = scores(t, s)
            End If
        Next t
    Next s

    CausalMask = out
End Function

' ---------------------------------------------------------------------------
' Zeilenweiser Softmax, numerisch stabil.
'
' Von jeder Zeile wird ihr Maximum abgezogen, bevor exponiert wird. Ohne das
' laufen die maskierten Werte (-1e30) in einen Ueberlauf statt gegen null.
' Der Boden bei -700 faengt genau die maskierten Eintraege ab: darunter waere
' Exp ein Unterlauf, so wird sauber die Null gesetzt, die auch das
' Referenzmodell dort stehen hat.
' Eingabe/Ausgabe: (nr, nc); jede Zeile summiert sich auf 1
Public Function SoftmaxRows(m() As Double) As Double()
    Dim i As Long, j As Long, nr As Long, nc As Long
    Dim mx As Double, s As Double, z As Double
    Dim out() As Double

    nr = UBound(m, 1)
    nc = UBound(m, 2)
    ReDim out(1 To nr, 1 To nc)

    For i = 1 To nr
        mx = m(i, 1)
        For j = 2 To nc
            If m(i, j) > mx Then mx = m(i, j)
        Next j

        s = 0#
        For j = 1 To nc
            z = m(i, j) - mx
            If z < EXP_FLOOR Then
                out(i, j) = 0#
            Else
                out(i, j) = Exp(z)
                s = s + out(i, j)
            End If
        Next j

        If s = 0# Then
            Err.Raise vbObjectError + 522, "Nn.SoftmaxRows", _
                      "Zeile " & i & " summiert sich auf null."
        End If
        For j = 1 To nc
            out(i, j) = out(i, j) / s
        Next j
    Next i

    SoftmaxRows = out
End Function

' ---------------------------------------------------------------------------
' Softmax ueber einen einzelnen Zeilenvektor, mit Temperatur.
' Eingabe:  logits (1, V), temperature > 0   Ausgabe: (1, V)
Public Function SoftmaxWithTemperature(logits() As Double, ByVal temperature As Double) As Double()
    Dim scaled() As Double
    Dim j As Long, nv As Long

    If temperature <= 0# Then
        Err.Raise vbObjectError + 523, "Nn.SoftmaxWithTemperature", _
                  "Temperatur muss groesser als null sein, war " & temperature
    End If

    nv = UBound(logits, 2)
    ReDim scaled(1 To 1, 1 To nv)
    For j = 1 To nv
        scaled(1, j) = logits(1, j) / temperature
    Next j

    SoftmaxWithTemperature = SoftmaxRows(scaled)
End Function
