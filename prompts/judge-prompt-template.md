### 1. Vorgaben und Regeln für die Übersetzung (System-Prompt des Modells)
Die Übersetzung wurde mit folgender Systemanweisung generiert:

```markdown
%SYSTEM_PROMPT%
```

---

### 2. Ausgangstext (Standardsprache)
```text
%ORIGINAL_TEXT%
```

---

### 3. Menschliche Referenzübersetzung (Ground Truth)
```text
%GROUND_TRUTH%
```

---

### 4. Zu bewertende Modellübersetzung (%MODEL_LABEL%)
```text
%TRANSLATION%
```

---

### Aufgabe für den Evaluator
Bewerte die vorgelegte Modellübersetzung anhand:
1. Der oben aufgeführten **Vorgaben und Regeln (System-Prompt des Modells)** (Regeltreue Leichte Sprache).
2. Des **Ausgangstexts in Standardsprache** (inhaltliche Vollständigkeit, keine Halluzinationen).
3. Der **menschlichen Referenzübersetzung** (Verständlichkeit, Natürlichkeit und direkter Qualitätsvergleich).

Antworte **ausschließlich** im geforderten JSON-Format gemäß System-Prompt.
