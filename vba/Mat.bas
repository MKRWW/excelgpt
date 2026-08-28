Attribute VB_Name = "Mat"
Option Explicit

' Matrix-Grundbausteine.
'
' Konvention fuer ALLE Matrizen in diesem Projekt:
'   - 1-basierte Double-Arrays der Form (Zeile, Spalte)
'   - Zeile = Zeitschritt t (1..T), Spalte = Merkmalsindex (1..C)
'   - dieselbe Anordnung wie in den Referenz-CSVs, nur um eins verschoben:
'     PyTorch zaehlt ab 0, VBA ab 1. Token-ID i steht also in Zeile i+1.
'
' Zellzugriff ausschliesslich blockweise: einmal Value2 in ein Variant-Array,
' danach nur noch im Speicher rechnen. Kein Range-Zugriff in einer Schleife.

Public Const MASK_VALUE As Double = -1E+30   ' Platzhalter fuer -unendlich

' ---------------------------------------------------------------------------
' Liest einen benannten Bereich als (Zeilen, Spalten)-Double-Matrix.
' Ein 1x1-Bereich liefert ueber Value2 einen Skalar statt eines Arrays --
' dieser Fall wird gesondert behandelt.
' Eingabe:  Name eines arbeitsmappenweiten Bereichs
' Ausgabe:  (nr, nc)
Public Function ReadNamed(ByVal rangeName As String) As Double()
    Dim rg As Range, v As Variant
    Dim nr As Long, nc As Long, i As Long, j As Long
    Dim out() As Double

    On Error GoTo Missing
    Set rg = ThisWorkbook.Names(rangeName).RefersToRange
    On Error GoTo 0

    nr = rg.Rows.Count
    nc = rg.Columns.Count
    ReDim out(1 To nr, 1 To nc)

    If nr = 1 And nc = 1 Then
        out(1, 1) = CDbl(rg.Value2)
    Else
        v = rg.Value2                       ' ein einziger Blockzugriff
        For j = 1 To nc
            For i = 1 To nr
                out(i, j) = CDbl(v(i, j))
            Next i
        Next j
    End If

    ReadNamed = out
    Exit Function
Missing:
    Err.Raise vbObjectError + 513, "Mat.ReadNamed", _
              "Benannter Bereich '" & rangeName & "' fehlt in der Arbeitsmappe."
End Function

' ---------------------------------------------------------------------------
' Schreibt eine Matrix in einem Zug auf ein Blatt, linke obere Ecke (r0, c0).
' Eingabe:  m (nr, nc)
Public Sub WriteBlock(ws As Worksheet, ByVal r0 As Long, ByVal c0 As Long, m() As Double)
    Dim nr As Long, nc As Long
    nr = UBound(m, 1)
    nc = UBound(m, 2)
    ws.Range(ws.Cells(r0, c0), ws.Cells(r0 + nr - 1, c0 + nc - 1)).Value2 = m
End Sub

' ---------------------------------------------------------------------------
' Neue Nullmatrix.
Public Function Zeros(ByVal nr As Long, ByVal nc As Long) As Double()
    Dim out() As Double
    ReDim out(1 To nr, 1 To nc)
    Zeros = out
End Function

' ---------------------------------------------------------------------------
' Matrixprodukt. a (m, k) mal b (k, n) ergibt (m, n).
'
' Die Schleifenreihenfolge ist j-p-i, nicht das uebliche i-j-p: VBA legt
' mehrdimensionale Arrays spaltenweise ab, der erste Index laeuft also im
' Speicher zusammenhaengend. In dieser Reihenfolge wandern sowohl a(i, p) als
' auch c(i, j) mit i durch den Speicher statt zu springen.
Public Function MatMul(a() As Double, b() As Double) As Double()
    Dim m As Long, k As Long, n As Long
    Dim i As Long, j As Long, p As Long
    Dim bpj As Double
    Dim c() As Double

    m = UBound(a, 1): k = UBound(a, 2)
    n = UBound(b, 2)
    If UBound(b, 1) <> k Then
        Err.Raise vbObjectError + 514, "Mat.MatMul", _
                  "Formen passen nicht: (" & m & "," & k & ") mal (" & _
                  UBound(b, 1) & "," & n & ")"
    End If

    ReDim c(1 To m, 1 To n)
    For j = 1 To n
        For p = 1 To k
            bpj = b(p, j)
            If bpj <> 0# Then
                For i = 1 To m
                    c(i, j) = c(i, j) + a(i, p) * bpj
                Next i
            End If
        Next p
    Next j

    MatMul = c
End Function

' ---------------------------------------------------------------------------
' y = x * W + b, mit b als Zeilenvektor (1, n), auf jede Zeile addiert.
' x (m, k), W (k, n), b (1, n) ergibt (m, n).
'
' Genau diese Form haben die exportierten Gewichte: (in, out). Deshalb wird
' hier nirgends transponiert.
Public Function MatMulAddBias(x() As Double, w() As Double, b() As Double) As Double()
    Dim out() As Double
    Dim i As Long, j As Long, m As Long, n As Long

    out = MatMul(x, w)
    m = UBound(out, 1)
    n = UBound(out, 2)
    If UBound(b, 2) <> n Then
        Err.Raise vbObjectError + 515, "Mat.MatMulAddBias", _
                  "Bias hat " & UBound(b, 2) & " Spalten, erwartet " & n
    End If

    For j = 1 To n
        For i = 1 To m
            out(i, j) = out(i, j) + b(1, j)
        Next i
    Next j

    MatMulAddBias = out
End Function

' ---------------------------------------------------------------------------
' Elementweise Summe zweier gleich geformter Matrizen -- die Residualaddition.
Public Function AddMat(a() As Double, b() As Double) As Double()
    Dim i As Long, j As Long, nr As Long, nc As Long
    Dim out() As Double

    nr = UBound(a, 1): nc = UBound(a, 2)
    If UBound(b, 1) <> nr Or UBound(b, 2) <> nc Then
        Err.Raise vbObjectError + 516, "Mat.AddMat", "Formen passen nicht."
    End If

    ReDim out(1 To nr, 1 To nc)
    For j = 1 To nc
        For i = 1 To nr
            out(i, j) = a(i, j) + b(i, j)
        Next i
    Next j

    AddMat = out
End Function

' ---------------------------------------------------------------------------
' Schneidet die Spalten c0..c1 heraus. src (nr, nc) ergibt (nr, c1 - c0 + 1).
' Wird gebraucht, um aus dem zusammengefassten QKV-Block die Anteile eines
' einzelnen Kopfes zu holen.
Public Function SliceCols(src() As Double, ByVal c0 As Long, ByVal c1 As Long) As Double()
    Dim i As Long, j As Long, nr As Long
    Dim out() As Double

    nr = UBound(src, 1)
    If c0 < 1 Or c1 > UBound(src, 2) Or c1 < c0 Then
        Err.Raise vbObjectError + 517, "Mat.SliceCols", _
                  "Spaltenbereich " & c0 & ".." & c1 & " liegt ausserhalb."
    End If

    ReDim out(1 To nr, 1 To c1 - c0 + 1)
    For j = c0 To c1
        For i = 1 To nr
            out(i, j - c0 + 1) = src(i, j)
        Next i
    Next j

    SliceCols = out
End Function

' ---------------------------------------------------------------------------
' Kopiert src (nr, k) in dst ab Spalte c0. Gegenstueck zu SliceCols: legt die
' Kopf-Ergebnisse in der Reihenfolge h = 0..n_head-1 nebeneinander.
Public Sub CopyIntoCols(dst() As Double, src() As Double, ByVal c0 As Long)
    Dim i As Long, j As Long
    For j = 1 To UBound(src, 2)
        For i = 1 To UBound(src, 1)
            dst(i, c0 + j - 1) = src(i, j)
        Next i
    Next j
End Sub

' ---------------------------------------------------------------------------
' Transponiert. a (m, n) ergibt (n, m). Gebraucht fuer Q mal K hoch T.
Public Function Transpose2(a() As Double) As Double()
    Dim i As Long, j As Long
    Dim out() As Double
    ReDim out(1 To UBound(a, 2), 1 To UBound(a, 1))
    For j = 1 To UBound(a, 2)
        For i = 1 To UBound(a, 1)
            out(j, i) = a(i, j)
        Next i
    Next j
    Transpose2 = out
End Function

' ---------------------------------------------------------------------------
' Multipliziert jedes Element mit einem Skalar -- die Attention-Skalierung.
Public Function ScaleMat(a() As Double, ByVal s As Double) As Double()
    Dim i As Long, j As Long
    Dim out() As Double
    ReDim out(1 To UBound(a, 1), 1 To UBound(a, 2))
    For j = 1 To UBound(a, 2)
        For i = 1 To UBound(a, 1)
            out(i, j) = a(i, j) * s
        Next i
    Next j
    ScaleMat = out
End Function
