Attribute VB_Name = "Sampler"
Option Explicit

' Ziehen des naechsten Tokens und die autoregressive Schleife.
'
' Temperatur teilt die Logits vor dem Softmax: kleine Werte schaerfen die
' Verteilung, grosse verwischen sie. Bei 0 waere es eine Division durch null,
' deshalb faengt Nn.SoftmaxWithTemperature das ab.

' ---------------------------------------------------------------------------
' Setzt den Zufallsstrom auf einen reproduzierbaren Anfang.
'
' Rnd mit negativem Argument setzt den Generator zurueck, erst danach waehlt
' Randomize den Startwert. Ohne das Zuruecksetzen liefert derselbe Startwert
' unterschiedliche Folgen, je nachdem was vorher lief.
Public Sub SeedRandom(ByVal seed As Long)
    Rnd -1
    Randomize seed
End Sub

' ---------------------------------------------------------------------------
' Zieht einen Index aus einer Wahrscheinlichkeitsverteilung.
' Eingabe: probs (1, V), Zeilensumme 1   Ausgabe: 0-basierte Token-ID
'
' Der Rueckfall auf das letzte Token faengt den Fall ab, dass die Summe durch
' Rundung minimal unter der gezogenen Zahl bleibt.
Public Function SampleFromProbs(probs() As Double) As Long
    Dim r As Double, acc As Double, j As Long

    r = Rnd()
    acc = 0#
    For j = 1 To UBound(probs, 2)
        acc = acc + probs(1, j)
        If r < acc Then
            SampleFromProbs = j - 1
            Exit Function
        End If
    Next j

    SampleFromProbs = UBound(probs, 2) - 1
End Function

' ---------------------------------------------------------------------------
' Schneidet die letzte Zeile einer Matrix als Zeilenvektor heraus.
' Eingabe: m (T, V)   Ausgabe: (1, V)
Public Function LastRow(m() As Double) As Double()
    Dim j As Long, nv As Long, nt As Long
    Dim out() As Double

    nt = UBound(m, 1)
    nv = UBound(m, 2)
    ReDim out(1 To 1, 1 To nv)
    For j = 1 To nv
        out(1, j) = m(nt, j)
    Next j

    LastRow = out
End Function

' ---------------------------------------------------------------------------
' Haengt ein Token an die Folge an und beschneidet den Kontext auf die letzten
' blockSize Eintraege -- das Positions-Embedding kennt nicht mehr Plaetze.
' Eingabe: tokens (1..n), neue ID   Ausgabe: (1..min(n+1, blockSize))
Public Function AppendCropped(tokens() As Long, ByVal newId As Long, _
                              ByVal blockSize As Long) As Long()
    Dim n As Long, keep As Long, i As Long, first As Long
    Dim out() As Long

    n = UBound(tokens)
    keep = n + 1
    If keep > blockSize Then keep = blockSize
    first = n + 1 - keep + 1          ' Index in der alten Folge, ab dem behalten wird

    ReDim out(1 To keep)
    For i = 1 To keep - 1
        out(i) = tokens(first + i - 1)
    Next i
    out(keep) = newId

    AppendCropped = out
End Function

' ---------------------------------------------------------------------------
' Autoregressive Erzeugung.
' Eingabe:  Prompt, Anzahl neuer Token, Temperatur, Startwert
' Ausgabe:  nur der erzeugte Text, ohne den Prompt
Public Function Generate(ByVal promptText As String, ByVal nTokens As Long, _
                         ByVal temperature As Double, ByVal seed As Long) As String
    Dim tokens() As Long, logits() As Double, probs() As Double
    Dim i As Long, nextId As Long
    Dim out As String

    Gpt.EnsureLoaded
    SeedRandom seed

    tokens = Gpt.EncodeText(promptText)
    out = ""

    For i = 1 To nTokens
        logits = Gpt.Forward(tokens)
        probs = Nn.SoftmaxWithTemperature(LastRow(logits), temperature)
        nextId = SampleFromProbs(probs)
        out = out & Gpt.Decode(nextId)
        tokens = AppendCropped(tokens, nextId, Gpt.BlockSize())
        DoEvents
    Next i

    Generate = out
End Function
