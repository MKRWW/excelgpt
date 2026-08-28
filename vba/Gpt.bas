Attribute VB_Name = "Gpt"
Option Explicit

' Der vollstaendige Vorwaertsdurchlauf.
'
' Reihenfolge je Block (Pre-LayerNorm):
'     x = x + attn(ln1(x))
'     x = x + mlp(ln2(x))
' danach logits = lm_head(ln_f(x)).
'
' Formen durchgehend (Zeile, Spalte) und 1-basiert:
'   x        (T, C)     T = Anzahl Token, C = 128
'   qkv      (T, 3C)    Spalten 1..128 = Q, 129..256 = K, 257..384 = V
'   je Kopf  (T, 32)    Kopf h belegt innerhalb jedes Drittels 32 Spalten
'   scores   (T, T)     Zeile = Abfrageposition, Spalte = Schluesselposition
'   logits   (T, V)     V = 65
'
' Die Gewichte werden einmal aus den benannten Bereichen in den Speicher
' geholt und dort gehalten. Sie bei jedem Token neu aus den Zellen zu lesen
' waere der sichere Weg in die Unbrauchbarkeit -- es sind 818241 Werte.

Private Type LayerWeights
    Ln1W() As Double
    Ln1B() As Double
    AttnW() As Double
    AttnB() As Double
    ProjW() As Double
    ProjB() As Double
    Ln2W() As Double
    Ln2B() As Double
    FcW() As Double
    FcB() As Double
    FcProjW() As Double
    FcProjB() As Double
End Type

Private mLoaded As Boolean
Private mNLayer As Long, mNHead As Long, mNEmbd As Long, mHeadDim As Long
Private mBlockSize As Long, mVocabSize As Long, mMlpHidden As Long
Private mWte() As Double, mWpe() As Double
Private mLnfW() As Double, mLnfB() As Double
Private mLmW() As Double, mLmB() As Double
Private mVocab() As Double
Private mLayers() As LayerWeights

' Ablage fuer den Mitschnitt der Zwischenergebnisse.
Private mTraceOn As Boolean
Private mTraceWs As Worksheet
Private mTraceRow As Long

' ---------------------------------------------------------------------------
' Laedt die Gewichte einmalig. Jeder weitere Aufruf kehrt sofort zurueck.
Public Sub EnsureLoaded()
    Dim l As Long, p As String

    If mLoaded Then Exit Sub

    mNLayer = CLng(Mat.ReadNamed("CFG_N_LAYER")(1, 1))
    mNHead = CLng(Mat.ReadNamed("CFG_N_HEAD")(1, 1))
    mNEmbd = CLng(Mat.ReadNamed("CFG_N_EMBD")(1, 1))
    mHeadDim = CLng(Mat.ReadNamed("CFG_HEAD_DIM")(1, 1))
    mBlockSize = CLng(Mat.ReadNamed("CFG_BLOCK_SIZE")(1, 1))
    mVocabSize = CLng(Mat.ReadNamed("CFG_VOCAB_SIZE")(1, 1))
    mMlpHidden = CLng(Mat.ReadNamed("CFG_MLP_HIDDEN")(1, 1))

    mWte = Mat.ReadNamed("WTE")
    mWpe = Mat.ReadNamed("WPE")
    mLnfW = Mat.ReadNamed("LNF_W")
    mLnfB = Mat.ReadNamed("LNF_B")
    mLmW = Mat.ReadNamed("LM_W")
    mLmB = Mat.ReadNamed("LM_B")
    mVocab = Mat.ReadNamed("VOCAB")

    ReDim mLayers(0 To mNLayer - 1)
    For l = 0 To mNLayer - 1
        p = "L" & l & "_"
        mLayers(l).Ln1W = Mat.ReadNamed(p & "LN1_W")
        mLayers(l).Ln1B = Mat.ReadNamed(p & "LN1_B")
        mLayers(l).AttnW = Mat.ReadNamed(p & "ATTN_W")
        mLayers(l).AttnB = Mat.ReadNamed(p & "ATTN_B")
        mLayers(l).ProjW = Mat.ReadNamed(p & "PROJ_W")
        mLayers(l).ProjB = Mat.ReadNamed(p & "PROJ_B")
        mLayers(l).Ln2W = Mat.ReadNamed(p & "LN2_W")
        mLayers(l).Ln2B = Mat.ReadNamed(p & "LN2_B")
        mLayers(l).FcW = Mat.ReadNamed(p & "FC_W")
        mLayers(l).FcB = Mat.ReadNamed(p & "FC_B")
        mLayers(l).FcProjW = Mat.ReadNamed(p & "FCPROJ_W")
        mLayers(l).FcProjB = Mat.ReadNamed(p & "FCPROJ_B")
    Next l

    mLoaded = True
End Sub

' Erzwingt ein Neuladen -- noetig, wenn die Gewichtsblaetter ersetzt wurden.
Public Sub ResetCache()
    mLoaded = False
End Sub

Public Function BlockSize() As Long
    EnsureLoaded
    BlockSize = mBlockSize
End Function

Public Function VocabSize() As Long
    EnsureLoaded
    VocabSize = mVocabSize
End Function

' ---------------------------------------------------------------------------
' Token-ID zu Zeichen. VOCAB haelt Codepoints, weil ein Apostroph als
' Zellwert von der Tabellenkalkulation verschluckt wuerde.
Public Function Decode(ByVal tokenId As Long) As String
    EnsureLoaded
    If tokenId < 0 Or tokenId >= mVocabSize Then
        Err.Raise vbObjectError + 540, "Gpt.Decode", "Token-ID " & tokenId & " liegt ausserhalb."
    End If
    Decode = ChrW$(CLng(mVocab(tokenId + 1, 1)))
End Function

' Zeichen zu Token-ID, oder -1, wenn es nicht im Vokabular steht.
Public Function Encode(ByVal ch As String) As Long
    Dim i As Long, cp As Long
    EnsureLoaded
    cp = AscW(ch)
    For i = 1 To mVocabSize
        If CLng(mVocab(i, 1)) = cp Then
            Encode = i - 1
            Exit Function
        End If
    Next i
    Encode = -1
End Function

' ---------------------------------------------------------------------------
' Text zu Token-Folge. Liefert ein 1-basiertes Long-Array mit 0-basierten IDs.
' Unbekannte Zeichen brechen ab -- stillschweigend zu ersetzen wuerde einen
' Prompt erzeugen, den niemand eingegeben hat.
Public Function EncodeText(ByVal s As String) As Long()
    Dim ids() As Long, i As Long, id As Long
    EnsureLoaded

    If Len(s) = 0 Then
        Err.Raise vbObjectError + 541, "Gpt.EncodeText", "Der Prompt ist leer."
    End If

    ReDim ids(1 To Len(s))
    For i = 1 To Len(s)
        id = Encode(Mid$(s, i, 1))
        If id < 0 Then
            Err.Raise vbObjectError + 542, "Gpt.EncodeText", _
                      "Zeichen '" & Mid$(s, i, 1) & "' (Position " & i & _
                      ") steht nicht im Vokabular."
        End If
        ids(i) = id
    Next i

    EncodeText = ids
End Function

' ---------------------------------------------------------------------------
' Mitschnitt der Zwischenergebnisse. Die Namen sind exakt die der
' Referenzdateien, damit sich beides ohne Uebersetzungstabelle vergleichen
' laesst.
Public Sub TraceBegin(ByVal sheetName As String)
    On Error GoTo Missing
    Set mTraceWs = ThisWorkbook.Worksheets(sheetName)
    On Error GoTo 0
    mTraceWs.Cells.Clear
    mTraceRow = 1
    mTraceOn = True
    Exit Sub
Missing:
    Err.Raise vbObjectError + 543, "Gpt.TraceBegin", "Blatt " & sheetName & " fehlt."
End Sub

Public Sub TraceEnd()
    mTraceOn = False
    Set mTraceWs = Nothing
End Sub

Private Sub Trace(ByVal keyName As String, m() As Double)
    Dim nr As Long, nc As Long
    If Not mTraceOn Then Exit Sub
    nr = UBound(m, 1)
    nc = UBound(m, 2)
    mTraceWs.Cells(mTraceRow, 1).Value2 = keyName & "  (" & nr & " x " & nc & ")"
    Mat.WriteBlock mTraceWs, mTraceRow + 1, 1, m
    mTraceRow = mTraceRow + nr + 2
End Sub

' ---------------------------------------------------------------------------
' Embedding-Lookup: Zeile tokens(t)+1 aus WTE plus Zeile t aus WPE.
' Die beiden Verschiebungen sind verschieden und deshalb einzeln kommentiert:
'   Token-ID 0 steht in WTE-Zeile 1  -> Index + 1
'   Position 0 steht in WPE-Zeile 1  -> t selbst, da t schon ab 1 laeuft
' Eingabe: tokens (1..T), 0-basierte IDs   Ausgabe: (T, C)
Private Function EmbedTokens(tokens() As Long) As Double()
    Dim t As Long, j As Long, nt As Long
    Dim x() As Double

    nt = UBound(tokens)
    If nt > mBlockSize Then
        Err.Raise vbObjectError + 544, "Gpt.EmbedTokens", _
                  "Kontext " & nt & " ueberschreitet das Fenster " & mBlockSize
    End If

    ReDim x(1 To nt, 1 To mNEmbd)
    For t = 1 To nt
        For j = 1 To mNEmbd
            x(t, j) = mWte(tokens(t) + 1, j) + mWpe(t, j)
        Next j
    Next t

    EmbedTokens = x
End Function

' ---------------------------------------------------------------------------
' Kausale Multi-Head-Attention eines Blocks.
' Eingabe: ln1 (T, C)   Ausgabe: (T, C) nach Ausgangsprojektion inklusive Bias
Private Function Attention(ln1() As Double, ByVal l As Long) As Double()
    Dim qkv() As Double, q() As Double, k() As Double, v() As Double
    Dim scores() As Double, masked() As Double, probs() As Double
    Dim headOut() As Double, concat() As Double
    Dim h As Long, col0 As Long, nt As Long
    Dim attnScale As Double

    nt = UBound(ln1, 1)
    attnScale = 1# / Sqr(CDbl(mHeadDim))
    qkv = Mat.MatMulAddBias(ln1, mLayers(l).AttnW, mLayers(l).AttnB)
    concat = Mat.Zeros(nt, mNEmbd)

    For h = 0 To mNHead - 1
        col0 = h * mHeadDim
        q = Mat.SliceCols(qkv, col0 + 1, col0 + mHeadDim)
        k = Mat.SliceCols(qkv, mNEmbd + col0 + 1, mNEmbd + col0 + mHeadDim)
        v = Mat.SliceCols(qkv, 2 * mNEmbd + col0 + 1, 2 * mNEmbd + col0 + mHeadDim)
        Trace "L" & l & "_11_q_h" & h, q
        Trace "L" & l & "_12_k_h" & h, k
        Trace "L" & l & "_13_v_h" & h, v

        scores = Mat.ScaleMat(Mat.MatMul(q, Mat.Transpose2(k)), attnScale)
        Trace "L" & l & "_14_scores_scaled_h" & h, scores

        masked = Nn.CausalMask(scores)
        Trace "L" & l & "_15_scores_masked_h" & h, masked

        probs = Nn.SoftmaxRows(masked)
        Trace "L" & l & "_16_attn_probs_h" & h, probs

        headOut = Mat.MatMul(probs, v)
        Trace "L" & l & "_17_head_out_h" & h, headOut

        Mat.CopyIntoCols concat, headOut, col0 + 1
    Next h

    Trace "L" & l & "_18_attn_concat", concat
    Attention = Mat.MatMulAddBias(concat, mLayers(l).ProjW, mLayers(l).ProjB)
End Function

' ---------------------------------------------------------------------------
' Ein vollstaendiger Block. Eingabe/Ausgabe (T, C).
Private Function Block(x() As Double, ByVal l As Long) As Double()
    Dim h1() As Double, attnOut() As Double, resid1() As Double
    Dim h2() As Double, fc() As Double, act() As Double, mlpOut() As Double
    Dim resid2() As Double

    h1 = Nn.LayerNorm(x, mLayers(l).Ln1W, mLayers(l).Ln1B)
    Trace "L" & l & "_10_ln1", h1

    attnOut = Attention(h1, l)
    Trace "L" & l & "_19_attn_proj", attnOut

    resid1 = Mat.AddMat(x, attnOut)
    Trace "L" & l & "_20_resid_post_attn", resid1

    h2 = Nn.LayerNorm(resid1, mLayers(l).Ln2W, mLayers(l).Ln2B)
    Trace "L" & l & "_30_ln2", h2

    fc = Mat.MatMulAddBias(h2, mLayers(l).FcW, mLayers(l).FcB)
    Trace "L" & l & "_31_fc", fc

    act = Nn.Gelu(fc)
    Trace "L" & l & "_32_gelu", act

    mlpOut = Mat.MatMulAddBias(act, mLayers(l).FcProjW, mLayers(l).FcProjB)
    Trace "L" & l & "_33_mlp_proj", mlpOut

    ' Ueber eine eigene Variable statt ueber den Funktionsnamen: der Name in
    ' einer Argumentliste liest sich fuer VBA als rekursiver Aufruf, nicht als
    ' Rueckgabewert.
    resid2 = Mat.AddMat(resid1, mlpOut)
    Trace "L" & l & "_34_resid_post_mlp", resid2
    Block = resid2
End Function

' ---------------------------------------------------------------------------
' Vollstaendiger Durchlauf ueber die gesamte Token-Folge.
' Eingabe: tokens (1..T), 0-basierte IDs   Ausgabe: logits (T, V)
Public Function Forward(tokens() As Long) As Double()
    Dim x() As Double, lnf() As Double, logits() As Double
    Dim l As Long, t As Long, nt As Long
    Dim tokMat() As Double, tokEmb() As Double, posEmb() As Double
    Dim j As Long

    EnsureLoaded
    nt = UBound(tokens)

    If mTraceOn Then
        ' Die drei Anteile einzeln, damit sich der Lookup pruefen laesst.
        ReDim tokMat(1 To nt, 1 To 1)
        ReDim tokEmb(1 To nt, 1 To mNEmbd)
        ReDim posEmb(1 To nt, 1 To mNEmbd)
        For t = 1 To nt
            tokMat(t, 1) = tokens(t)
            For j = 1 To mNEmbd
                tokEmb(t, j) = mWte(tokens(t) + 1, j)
                posEmb(t, j) = mWpe(t, j)
            Next j
        Next t
        Trace "00_tokens", tokMat
        Trace "01_tok_emb", tokEmb
        Trace "02_pos_emb", posEmb
    End If

    x = EmbedTokens(tokens)
    Trace "03_x_input", x

    For l = 0 To mNLayer - 1
        x = Block(x, l)
    Next l

    lnf = Nn.LayerNorm(x, mLnfW, mLnfB)
    Trace "90_ln_f", lnf

    logits = Mat.MatMulAddBias(lnf, mLmW, mLmB)
    Trace "91_logits", logits
    Forward = logits
End Function

' ---------------------------------------------------------------------------
' Durchlauf fuer einen Text, mit Mitschnitt aller Zwischenergebnisse.
' Genau der Einstieg, gegen den die Referenzdaten gestellt werden.
Public Sub RunTrace(ByVal promptText As String, ByVal sheetName As String)
    Dim tokens() As Long, logits() As Double
    Dim lastRow() As Double, probs() As Double
    Dim j As Long, nv As Long

    EnsureLoaded
    tokens = EncodeText(promptText)

    TraceBegin sheetName
    logits = Forward(tokens)

    nv = UBound(logits, 2)
    ReDim lastRow(1 To 1, 1 To nv)
    For j = 1 To nv
        lastRow(1, j) = logits(UBound(logits, 1), j)
    Next j
    Trace "92_logits_last", lastRow

    probs = Nn.SoftmaxRows(lastRow)
    Trace "93_probs_last_temp1", probs

    TraceEnd
End Sub
