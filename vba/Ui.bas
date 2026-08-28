Attribute VB_Name = "Ui"
Option Explicit

' Das Bedienpult auf dem Blatt 00_LLM.
'
' Das gesamte Layout entsteht in SetupSheet, nicht von Hand -- damit ist es
' versionierbar und nach jedem Neubau identisch. Wer etwas verschiebt, aendert
' den Code, nicht die Zellen.
'
' Alle Spalten sind schmal und gleich breit, damit der Heatmap-Bereich ein
' quadratisches Raster ergibt. Beschriftungen und Eingabefelder ueberspannen
' deshalb jeweils mehrere Spalten.
'
' Angesprochen wird auch hier ueber benannte Bereiche, nie ueber Zelladressen:
'   UI_PROMPT  UI_TEMP  UI_TOKENS  UI_HEAD  UI_SEED  UI_STATUS  UI_OUTPUT
'   UI_HEATMAP

Private Const SHEET_NAME As String = "00_LLM"
Private Const GRID_COLS As Long = 64          ' = block_size, Spalten A..BL
Private Const COL_WIDTH As Double = 2#
Private Const GRID_ROW_HEIGHT As Double = 14.25   ' ergibt mit COL_WIDTH ~quadratisch
Private Const GRID_TOP As Long = 24
Private Const FIELD_COL As Long = 7           ' Eingaben beginnen in Spalte G
Private Const FIELD_END As Long = 32          ' und reichen bis AF

Private Const CLR_INK As Long = 3355443       ' dunkles Grau, RGB(51,51,51)
Private Const CLR_LABEL As Long = 8421504     ' mittleres Grau
Private Const CLR_FIELD As Long = 15921906    ' sehr helles Blaugrau
Private Const CLR_FRAME As Long = 12105912    ' Rahmen der Eingabefelder
Private Const CLR_HEAT_HI As Long = 7954483   ' Blau fuer den oberen Skalenwert

' ---------------------------------------------------------------------------
Private Function Sheet() As Worksheet
    On Error GoTo Missing
    Set Sheet = ThisWorkbook.Worksheets(SHEET_NAME)
    Exit Function
Missing:
    Err.Raise vbObjectError + 560, "Ui", "Blatt " & SHEET_NAME & " fehlt."
End Function

' ---------------------------------------------------------------------------
' Beschriftung links, ueber die Spalten A bis F.
Private Sub Label(ws As Worksheet, ByVal r As Long, ByVal text As String)
    With ws.Range(ws.Cells(r, 1), ws.Cells(r, 6))
        .Merge
        .Value2 = text
        .Font.Color = CLR_LABEL
        .Font.Size = 10
        .HorizontalAlignment = xlLeft
        .VerticalAlignment = xlCenter
    End With
End Sub

' Eingabefeld: gerahmt und hinterlegt, damit es als Eingabe erkennbar ist.
Private Function Field(ws As Worksheet, ByVal r As Long, ByVal c1 As Long, _
                       ByVal c2 As Long, ByVal rangeName As String) As Range
    Dim rg As Range
    Set rg = ws.Range(ws.Cells(r, c1), ws.Cells(r, c2))
    With rg
        .Merge
        .Interior.Color = CLR_FIELD
        .BorderAround Weight:=xlThin, Color:=CLR_FRAME
        .Font.Color = CLR_INK
        .Font.Size = 11
        .HorizontalAlignment = xlLeft
        .VerticalAlignment = xlCenter
        .IndentLevel = 1
    End With
    ThisWorkbook.Names.Add name:=rangeName, RefersTo:=rg
    Set Field = rg
End Function

' ---------------------------------------------------------------------------
' Baut das Bedienpult vollstaendig neu auf.
' Liefert eine leere Zeichenkette bei Erfolg, sonst den Fehlertext. Ein
' unbehandelter Fehler wuerde in einer unsichtbaren Instanz einen Dialog
' oeffnen, auf den niemand klicken kann -- der Aufruf haengt dann, statt zu
' scheitern.
Public Function SetupSheetSafe() As String
    On Error GoTo Failed
    SetupSheet
    SetupSheetSafe = ""
    Exit Function
Failed:
    Application.ScreenUpdating = True
    SetupSheetSafe = "Fehler " & Err.Number & ": " & Err.Description
End Function

Public Sub SetupSheet()
    Dim ws As Worksheet, rg As Range, shp As Shape
    Dim i As Long

    Set ws = Sheet()

    Application.ScreenUpdating = False

    ' Rueckwaerts ueber den Index, nicht mit For Each: beim Loeschen aus einer
    ' Collection waehrend der Iteration ueberspringt VBA Eintraege.
    For i = ws.Shapes.Count To 1 Step -1
        ws.Shapes(i).Delete
    Next i
    For i = ThisWorkbook.Names.Count To 1 Step -1
        If Left$(ThisWorkbook.Names(i).name, 3) = "UI_" Then
            ThisWorkbook.Names(i).Delete
        End If
    Next i

    ws.Cells.Clear
    ws.Cells.ClearFormats
    ws.Cells.Font.name = "Consolas"
    ws.Cells.Font.Size = 10

    ' Ein durchgehend schmales Raster. Nur so wird die Heatmap quadratisch.
    ws.Columns(1).Resize(, GRID_COLS + 4).ColumnWidth = COL_WIDTH
    ws.Rows(GRID_TOP).Resize(GRID_COLS).RowHeight = GRID_ROW_HEIGHT

    ' --- Kopfzeile ---
    With ws.Range(ws.Cells(2, 1), ws.Cells(2, 24))
        .Merge
        .Value2 = "excelgpt"
        .Font.Size = 16
        .Font.Bold = True
        .Font.Color = CLR_INK
    End With
    With ws.Range(ws.Cells(3, 1), ws.Cells(3, 40))
        .Merge
        .Value2 = "Ein Sprachmodell, das in Zellen liegt und in Makrocode rechnet."
        .Font.Color = CLR_LABEL
    End With

    ' --- Eingaben ---
    ws.Rows(5).RowHeight = 22
    Label ws, 5, "Prompt"
    Field(ws, 5, FIELD_COL, FIELD_END, "UI_PROMPT").Value2 = "ROMEO:"

    Label ws, 7, "Temperatur"
    Field(ws, 7, FIELD_COL, FIELD_COL + 5, "UI_TEMP").Value2 = 0.8
    ws.Cells(7, FIELD_COL + 7).Value2 = "klein = braver, gross = wilder"
    ws.Cells(7, FIELD_COL + 7).Font.Color = CLR_LABEL

    Label ws, 8, "Token"
    Field(ws, 8, FIELD_COL, FIELD_COL + 5, "UI_TOKENS").Value2 = 120
    ws.Cells(8, FIELD_COL + 7).Value2 = "Anzahl der Zeichen, die erzeugt werden"
    ws.Cells(8, FIELD_COL + 7).Font.Color = CLR_LABEL

    Label ws, 9, "Kopf"
    Field(ws, 9, FIELD_COL, FIELD_COL + 5, "UI_HEAD").Value2 = 0
    ws.Cells(9, FIELD_COL + 7).Value2 = "welcher Aufmerksamkeitskopf unten gezeigt wird, 0 bis 3"
    ws.Cells(9, FIELD_COL + 7).Font.Color = CLR_LABEL

    Label ws, 10, "Startwert"
    Field(ws, 10, FIELD_COL, FIELD_COL + 5, "UI_SEED").Value2 = 1337
    ws.Cells(10, FIELD_COL + 7).Value2 = "gleicher Startwert und gleiche Eingaben ergeben denselben Text"
    ws.Cells(10, FIELD_COL + 7).Font.Color = CLR_LABEL

    ' --- Knopf ---
    Set shp = ws.Shapes.AddShape(msoShapeRoundedRectangle, _
                                 ws.Cells(5, FIELD_END + 3).Left, _
                                 ws.Cells(5, 1).Top, 120, 46)
    With shp
        .name = "btnGenerate"
        .Fill.ForeColor.RGB = CLR_HEAT_HI
        .Line.Visible = msoFalse
        .TextFrame2.TextRange.text = "Generate"
        .TextFrame2.TextRange.Font.Size = 12
        .TextFrame2.TextRange.Font.Bold = msoTrue
        .TextFrame2.TextRange.Font.Fill.ForeColor.RGB = RGB(255, 255, 255)
        .OnAction = "Ui.Generate"
    End With

    ' --- Status ---
    Label ws, 12, "Status"
    With ws.Range(ws.Cells(12, FIELD_COL), ws.Cells(12, FIELD_END))
        .Merge
        .Value2 = "bereit"
        .Font.Color = CLR_INK
    End With
    ThisWorkbook.Names.Add name:="UI_STATUS", _
        RefersTo:=ws.Range(ws.Cells(12, FIELD_COL), ws.Cells(12, FIELD_END))

    ' --- Ausgabe ---
    Label ws, 14, "Ausgabe"
    Set rg = ws.Range(ws.Cells(14, FIELD_COL), ws.Cells(20, FIELD_END))
    With rg
        .Merge
        .WrapText = True
        .HorizontalAlignment = xlLeft
        .VerticalAlignment = xlTop
        .Font.Color = CLR_INK
        .BorderAround Weight:=xlThin, Color:=CLR_FRAME
    End With
    ThisWorkbook.Names.Add name:="UI_OUTPUT", RefersTo:=rg

    ' --- Heatmap ---
    With ws.Range(ws.Cells(22, 1), ws.Cells(22, 40))
        .Merge
        .Value2 = "Aufmerksamkeit im letzten Layer  --  Zeile = das Zeichen, das dran ist, Spalte = worauf es zurueckschaut"
        .Font.Color = CLR_LABEL
    End With

    Set rg = ws.Range(ws.Cells(GRID_TOP, 1), _
                      ws.Cells(GRID_TOP + GRID_COLS - 1, GRID_COLS))
    rg.NumberFormat = ";;;"        ' Werte rechnen, aber nicht anzeigen
    rg.Interior.Color = RGB(255, 255, 255)
    ThisWorkbook.Names.Add name:="UI_HEATMAP", RefersTo:=rg
    ApplyColorScale rg

    ' Gitternetz aus. Ueber das Fenster der Arbeitsmappe statt ueber
    ' ActiveWindow: beim Bau aus einem unsichtbaren Vorgang heraus gibt es kein
    ' aktives Fenster, ein Fenster der Mappe aber sehr wohl. Die Einstellung
    ' wird mit der Datei gespeichert und gilt beim naechsten Oeffnen.
    On Error Resume Next
    ThisWorkbook.Windows(1).DisplayGridlines = False
    On Error GoTo 0

    Application.ScreenUpdating = True
End Sub

' ---------------------------------------------------------------------------
' Farbskala als bedingte Formatierung: weiss fuer null, blau fuer den
' groessten Wert im Bereich. Die Skala passt sich selbst an, sonst waere bei
' Aufmerksamkeitswerten, die meist klein sind, alles blass.
Private Sub ApplyColorScale(rg As Range)
    Dim cs As ColorScale
    rg.FormatConditions.Delete
    Set cs = rg.FormatConditions.AddColorScale(ColorScaleType:=2)
    With cs.ColorScaleCriteria(1)
        .Type = xlConditionValueLowestValue
        .FormatColor.Color = RGB(255, 255, 255)
    End With
    With cs.ColorScaleCriteria(2)
        .Type = xlConditionValueHighestValue
        .FormatColor.Color = CLR_HEAT_HI
    End With
End Sub

' ---------------------------------------------------------------------------
' Ansicht herrichten: Gitternetz aus, Bedienpult vorn.
'
' Wird beim Oeffnen der Arbeitsmappe aufgerufen. Die Einstellung hier statt nur
' in SetupSheet zu setzen ist noetig, weil das Layout in einer unsichtbaren
' Instanz gebaut wird -- deren Fenster ist nicht das, in dem spaeter jemand
' sitzt, und die Einstellung haengt am Fenster.
Public Sub ApplyView()
    Dim w As Window
    On Error Resume Next
    For Each w In ThisWorkbook.Windows
        w.DisplayGridlines = False
    Next w
    ThisWorkbook.Worksheets(SHEET_NAME).Activate
    On Error GoTo 0
End Sub

' ---------------------------------------------------------------------------
' Die linke obere Zelle eines Bedienelements.
'
' Der Umweg ist noetig, weil die Felder verbundene Bereiche sind: deren Value2
' liefert ein ganzes Array statt eines Einzelwerts, und jede Umwandlung daraus
' scheitert mit "Typen unvertraeglich".
Private Function Ctl(ByVal rangeName As String) As Range
    Set Ctl = ThisWorkbook.Names(rangeName).RefersToRange.Cells(1, 1)
End Function

Private Sub Status(ByVal text As String)
    Ctl("UI_STATUS").Value2 = text
End Sub

' Schreibt die gemerkte Aufmerksamkeitsmatrix in einem Zug in das Raster.
' Ausserhalb der belegten T mal T bleibt es null, damit die Farbskala dort
' weiss zeichnet statt alte Werte stehen zu lassen.
Private Sub DrawHeatmap()
    Dim a() As Double, grid() As Double
    Dim i As Long, j As Long, n As Long

    If Not Gpt.HasAttention() Then Exit Sub
    a = Gpt.LastAttention()
    n = UBound(a, 1)
    If n > GRID_COLS Then n = GRID_COLS

    ReDim grid(1 To GRID_COLS, 1 To GRID_COLS)
    For j = 1 To n
        For i = 1 To n
            grid(i, j) = a(i, j)
        Next i
    Next j

    ThisWorkbook.Names("UI_HEATMAP").RefersToRange.Value2 = grid
End Sub

' ---------------------------------------------------------------------------
' Liest die Eingaben, prueft sie und erzeugt Text. Nach jedem Token werden
' Ausgabe, Status und Heatmap aufgefrischt -- deshalb laeuft die Schleife hier
' und nicht in Sampler.Generate, das bewusst ohne Blattverkehr auskommt.
Public Sub Generate()
    Dim ws As Worksheet
    Dim promptText As String, produced As String
    Dim temperature As Double
    Dim nTokens As Long, headIdx As Long, seed As Long
    Dim tokens() As Long, logits() As Double, probs() As Double
    Dim i As Long, nextId As Long
    Dim started As Double

    Set ws = Sheet()
    On Error GoTo Failed

    promptText = CStr(Ctl("UI_PROMPT").Value2)
    temperature = CDbl(Ctl("UI_TEMP").Value2)
    nTokens = CLng(Ctl("UI_TOKENS").Value2)
    headIdx = CLng(Ctl("UI_HEAD").Value2)
    seed = CLng(Ctl("UI_SEED").Value2)

    If Len(promptText) = 0 Then
        Status "Der Prompt ist leer."
        Exit Sub
    End If
    If temperature <= 0# Then
        Status "Die Temperatur muss groesser als null sein."
        Exit Sub
    End If
    If nTokens < 1 Then
        Status "Die Anzahl Token muss mindestens eins sein."
        Exit Sub
    End If
    If headIdx < 0 Or headIdx >= Gpt.HeadCount() Then
        Status "Kopf muss zwischen 0 und " & (Gpt.HeadCount() - 1) & " liegen."
        Exit Sub
    End If

    Status "Gewichte werden geladen ..."
    Gpt.EnsureLoaded
    Ctl("UI_OUTPUT").Value2 = ""
    ThisWorkbook.Names("UI_HEATMAP").RefersToRange.ClearContents

    Gpt.CaptureAttention Gpt.LayerCount() - 1, headIdx
    Gpt.SetProgressCell Ctl("UI_STATUS")
    Sampler.SeedRandom seed

    tokens = Gpt.EncodeText(promptText)
    produced = ""
    started = Timer

    For i = 1 To nTokens
        logits = Gpt.Forward(tokens)
        probs = Nn.SoftmaxWithTemperature(Sampler.LastRow(logits), temperature)
        nextId = Sampler.SampleFromProbs(probs)
        produced = produced & Gpt.Decode(nextId)

        Ctl("UI_OUTPUT").Value2 = produced
        DrawHeatmap
        Status "Token " & i & " von " & nTokens & _
               "   (" & Format$(Timer - started, "0.0") & " s)"

        tokens = Sampler.AppendCropped(tokens, nextId, Gpt.BlockSize())
        DoEvents
    Next i

    Gpt.ClearProgressCell
    Gpt.CaptureOff
    Status "fertig -- " & nTokens & " Token in " & _
           Format$(Timer - started, "0.0") & " s"
    Exit Sub

Failed:
    Gpt.ClearProgressCell
    Gpt.CaptureOff
    Status "Abbruch: " & Err.Description
End Sub
