# Wie man mit diesem Modell redet

Kurzfassung: **gar nicht.** Man gibt ihm einen Textanfang, und es schreibt weiter.
Wer es um etwas bittet, bekommt keine Antwort, sondern eine Fortsetzung seiner Bitte.

Das ist kein Mangel, den man wegkonfigurieren koennte, sondern die Natur der Sache.
Dieses Dokument erklaert, woran das liegt, und was stattdessen funktioniert.

## Womit man es zu tun hat

| | |
|---|---|
| Parameter | 818.241 |
| Trainingstext | Tiny Shakespeare, 1.115.394 Zeichen |
| Einheit | ein **Zeichen**, nicht ein Wort |
| Kontextfenster | 64 Zeichen |
| Gelernte Aufgabe | das naechste Zeichen vorhersagen |

Zum Groessenvergleich: ein heutiges Sprachmodell hat grob das Hunderttausendfache an
Parametern und hat einen erheblichen Teil des Internets gelesen. Dieses hier hat
ausschliesslich Theaterstuecke gesehen, und zwar Zeichen fuer Zeichen.

Es wurde nie darauf trainiert, Anweisungen zu befolgen. Es gibt in seiner Welt keine
Fragen, keine Aufgaben, keinen Dialog mit einem Benutzer. Es gibt nur: hier steht Text,
welches Zeichen kommt als naechstes.

## Warum Anweisungen nicht funktionieren

Ein Blick in den Trainingstext erklaert es besser als jede Theorie:

| Zeichenfolge | wie oft im Trainingstext |
|---|---|
| `Make a` | 2 |
| `rhyme` | 3 |
| `sheep` | 16 |
| `about` | 83 |

Der Prompt *"Make a short rhyme about a sheep"* besteht also aus Bausteinen, die das
Modell kaum kennt, in einer Anordnung, die es nie gesehen hat. Es tut das Einzige, was
es kann — weiterschreiben:

```
Make a short rhyme about a sheep-bell your lands;
Less of speak needful not his love down,
```

Es hat `sheep` zu `sheep-bell` vervollstaendigt und ist dann in den Versduktus gekippt.
Die Aufforderung wurde nicht abgelehnt, sie wurde nie als solche wahrgenommen.

## Was funktioniert

Alles, was aussieht wie der Anfang einer Zeile in einem Theaterstueck. Die mit Abstand
staerkste Struktur im Text ist die **Sprecher-Anrede**: ein Name in Grossbuchstaben,
Doppelpunkt, Zeilenumbruch, dann Blankvers. Die haeufigsten:

| Anrede | wie oft |
|---|---|
| `GLOUCESTER:` | 229 |
| `DUKE VINCENTIO:` | 193 |
| `ROMEO:` | 163 |
| `MENENIUS:` | 162 |
| `PETRUCHIO:` | 158 |
| `CORIOLANUS:` | 149 |
| `KING RICHARD III:` | 138 |
| `ISABELLA:` | 129 |
| `JULIET:` | 125 |

Damit als Prompt weiss das Modell sofort, wohin die Reise geht:

```
ROMEO:
Shall you do revenge for defend of this is here.

MENENIUS:
Ay, love a senator.

LADY GUE:
For alre is the wor
```

Weitere brauchbare Anfaenge:

- **Anrede plus Ausrufanfang** — `JULIET:` und darunter `O `
- **mitten im Vers einsteigen** — `To be, or not to be`
- **haeufige Zeilenanfaenge** — `And`, `I`, `The`, `To`, `That`, `But`, `My`, `What`

Fuer einen Zeilenumbruch in der Prompt-Zelle: **Alt+Enter**. Notwendig ist er nicht, das
Modell setzt ihn nach einer Anrede meist von selbst.

## Die Temperatur

Der Regler mit der groessten Wirkung. Er teilt die Logits vor dem Softmax: kleine Werte
schaerfen die Verteilung, grosse verwischen sie. Dreimal derselbe Prompt `ROMEO:` und
derselbe Startwert, nur die Temperatur unterscheidet sich.

**0,3 — geordnet, aber floskelhaft.** Es greift immer zum wahrscheinlichsten Zeichen und
landet in ausgetretenen Wendungen:

```
What I have been to my life, and heart thou hast thou
art once of the conscience the world of the day.
```

**0,8 — die brauchbare Mitte.** Satzbau haelt, Figurenwechsel funktionieren, Woerter sind
ueberwiegend echt. Siehe das Beispiel weiter oben.

**1,4 — Zerfall.** Die Verteilung ist so flach, dass unwahrscheinliche Zeichen
durchkommen:

```
Redom
year of your egener: once master's son, open.

GRUMIO:
Whereup a she: his long
town delike?.

ISABELLUY:
```

Empfehlung: zwischen **0,6 und 0,9** bleiben. Darunter wird es langweilig, darueber
unlesbar.

## Der Startwert

Gleicher Startwert, gleicher Prompt, gleiche Temperatur ergeben denselben Text. Wer eine
Formulierung wiederfinden will, merkt sich den Startwert. Wer Abwechslung will, aendert
ihn.

## Was man an den erfundenen Namen sieht

Im Beispiel oben stehen `LADY GUE:` und `ISABELLUY:`. Beide Figuren gibt es nicht.

Das ist der lehrreichste Teil: das Modell hat nie eine Liste von Namen gelernt. Es hat
gelernt, dass nach einem Zeilenumbruch manchmal Grossbuchstaben folgen, dann ein
Doppelpunkt, dann wieder ein Umbruch. Es baut die **Form** eines Namens nach, Buchstabe
fuer Buchstabe, ohne zu wissen, dass ein Name etwas bezeichnet. `ISABELLUY` liegt einen
Fehltritt neben `ISABELLA`, die 129 mal im Text steht.

Wer wissen will, was das Modell in einem bestimmten Moment ansieht, schaut in die
Heatmap: Zeile ist das Zeichen, das gerade dran ist, Spalte ist das, worauf es
zurueckgreift. Mitten im Wort sind es die letzten Buchstaben, nach einem Zeilenumbruch
der Zeilenanfang.

## Erwartungen

Es kommt kein Sinn heraus. Es kommt etwas heraus, das **klingt** wie Shakespeare —
Versrhythmus, Figurenwechsel, Interpunktion an plausiblen Stellen, gelegentlich eine
Wendung, die fast eine Bedeutung hat. Das ist die ganze Vorstellung, und sie ist fuer
818.241 Zahlen in Tabellenzellen bemerkenswert genug.
