Attribute VB_Name = "Probe"
Option Explicit

' Pruefeinstiege fuer die Bausteine.
'
' Jede Prozedur hier liest ihre Eingabe aus einem Zellbereich des Blattes
' 98_Probe, ruft genau einen Baustein auf und schreibt das Ergebnis zurueck.
' So laesst sich jeder Baustein einzeln gegen die Referenzdaten stellen,
' bevor er in den Stapel eingebaut wird -- statt am Ende zu raten, welche
' der acht Stufen das Rauschen erzeugt.
'
' Die Adressen kommen als Zeichenketten von aussen, damit die Pruefung die
' Groessen bestimmt und nicht der VBA-Code.

Private Const PROBE_SHEET As String = "98_Probe"

' ---------------------------------------------------------------------------
Private Function ProbeSheet() As Worksheet
    On Error GoTo Missing
    Set ProbeSheet = ThisWorkbook.Worksheets(PROBE_SHEET)
    Exit Function
Missing:
    Err.Raise vbObjectError + 530, "Probe", "Blatt " & PROBE_SHEET & " fehlt."
End Function

' ---------------------------------------------------------------------------
' Liest einen Bereich des Pruefblattes als Matrix. Ein 1x1-Bereich kommt ueber
' Value2 als Skalar zurueck und wird gesondert behandelt.
Private Function ReadAddr(ByVal addr As String) As Double()
    Dim rg As Range, v As Variant
    Dim i As Long, j As Long, nr As Long, nc As Long
    Dim out() As Double

    Set rg = ProbeSheet().Range(addr)
    nr = rg.Rows.Count
    nc = rg.Columns.Count
    ReDim out(1 To nr, 1 To nc)

    If nr = 1 And nc = 1 Then
        out(1, 1) = CDbl(rg.Value2)
    Else
        v = rg.Value2
        For j = 1 To nc
            For i = 1 To nr
                out(i, j) = CDbl(v(i, j))
            Next i
        Next j
    End If

    ReadAddr = out
End Function

' ---------------------------------------------------------------------------
' Schreibt eine Matrix ab der Zelle addr in einem Zug.
Private Sub WriteAt(ByVal addr As String, m() As Double)
    Dim ws As Worksheet, r0 As Long, c0 As Long
    Set ws = ProbeSheet()
    r0 = ws.Range(addr).Row
    c0 = ws.Range(addr).Column
    Mat.WriteBlock ws, r0, c0, m
End Sub

' ---------------------------------------------------------------------------
' Matrixprodukt zweier Bereiche.
Public Sub ProbeMatMul(ByVal aAddr As String, ByVal bAddr As String, ByVal outAddr As String)
    WriteAt outAddr, Mat.MatMul(ReadAddr(aAddr), ReadAddr(bAddr))
End Sub

' ---------------------------------------------------------------------------
' y = x * W + b, mit Gewicht und Bias aus benannten Bereichen der Arbeitsmappe.
Public Sub ProbeMatMulAddBias(ByVal xAddr As String, ByVal wName As String, _
                              ByVal bName As String, ByVal outAddr As String)
    WriteAt outAddr, Mat.MatMulAddBias(ReadAddr(xAddr), _
                                       Mat.ReadNamed(wName), Mat.ReadNamed(bName))
End Sub

' ---------------------------------------------------------------------------
' LayerNorm mit Gewicht und Bias aus benannten Bereichen.
Public Sub ProbeLayerNorm(ByVal xAddr As String, ByVal wName As String, _
                          ByVal bName As String, ByVal outAddr As String)
    WriteAt outAddr, Nn.LayerNorm(ReadAddr(xAddr), _
                                  Mat.ReadNamed(wName), Mat.ReadNamed(bName))
End Sub

' ---------------------------------------------------------------------------
Public Sub ProbeGelu(ByVal xAddr As String, ByVal outAddr As String)
    WriteAt outAddr, Nn.Gelu(ReadAddr(xAddr))
End Sub

' ---------------------------------------------------------------------------
Public Sub ProbeSoftmaxRows(ByVal mAddr As String, ByVal outAddr As String)
    WriteAt outAddr, Nn.SoftmaxRows(ReadAddr(mAddr))
End Sub

' ---------------------------------------------------------------------------
Public Sub ProbeCausalMask(ByVal mAddr As String, ByVal outAddr As String)
    WriteAt outAddr, Nn.CausalMask(ReadAddr(mAddr))
End Sub

' ---------------------------------------------------------------------------
' Liest einen benannten Bereich und schreibt ihn unveraendert zurueck --
' prueft den Weg Gewicht raus, Gewicht rein, ohne Rechnung dazwischen.
Public Sub ProbeReadNamed(ByVal rangeName As String, ByVal outAddr As String)
    WriteAt outAddr, Mat.ReadNamed(rangeName)
End Sub

' ---------------------------------------------------------------------------
' Einzelwert-Einstiege, damit sich Randfaelle ohne Blattverkehr pruefen lassen.
Public Function ProbeTanh(ByVal z As Double) As Double
    ProbeTanh = Nn.Tanh(z)
End Function

Public Function ProbeGeluScalar(ByVal x As Double) As Double
    Dim m() As Double, r() As Double
    ReDim m(1 To 1, 1 To 1)
    m(1, 1) = x
    r = Nn.Gelu(m)
    ProbeGeluScalar = r(1, 1)
End Function
