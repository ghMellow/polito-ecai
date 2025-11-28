## **Esercizio 1: Depthwise Separable Convolutions**
**File:** `7.1 - Lab4 - Training with DS-CNN.ipynb`

**Obiettivo:** Implementare e addestrare una CNN con convoluzioni depthwise separable

**Passi:**
1. Clonare il notebook `6.1 - Lab3 - KWS Training`
2. Implementare l'architettura DS-CNN secondo la tabella (Conv2D → DConv+SConv ripetuti → GAP → Linear)
3. Addestrare con: batch size 32, Adam optimizer, LR 0.001, 2000 steps
4. Sperimentare variando:
   - Frame length STFT: 10-50ms (anche 8, 16, 32ms)
   - Overlap: 0%, 25%, 50%, 75%
   - Mel bins: 10-40
   - Frequenze mel: lower 20-80Hz, upper 2000-8000Hz
   - MFCC: 10-40
5. Valutare accuratezza e dimensioni ONNX
6. Deployare su Raspberry Pi e misurare latenza


## **Sequenza di test (partendo dal baseline attuale):**

**Baseline (punto di partenza):**
```python
frame_length_in_s: 0.04   # 40ms
frame_step_in_s: 0.02     # 50% overlap
n_mels: 40
f_min: 0
f_max: 8000
n_mfcc: 40
```

**NOTA:** `n_mfcc` deve essere `<= n_mels` sempre!

---

### **Test 1: Ridurre n_mfcc**
```python
n_mfcc: 10
```

### **Test 2: Ridurre n_mels** (richiede anche n_mfcc <= n_mels)
```python
n_mels: 10
n_mfcc: 10
```

### No vantaggi tengo:
```python
n_mels: 40
n_mfcc: 40
```
---

### **Test 3: f_min = 20 Hz**
```python
f_min: 20
```

### **Test 4: f_min = 80 Hz**
```python
f_min: 80
```
*(Mantieni il migliore tra 0, 20, 80)*
0 
**Risultato Test 3-4:** Accuratezza uguale per f_min = 0, 20, 80
**Configurazione scelta:** 
```python
f_min: 20  # elimina rumore sotto 20Hz senza perdere accuratezza
```

---

### **Test 5: f_max = 2000 Hz**
```python
f_max: 2000
```

### **Test 6: f_max = 4000 Hz**
```python
f_max: 4000
```
*(Mantieni il migliore tra 2000, 4000, 8000)*

4000 non perde accuracy tengo
---

### **Test 7: Frame length = 8ms**
```python
frame_length_in_s: 0.008
frame_step_in_s: 0.004  # mantieni 50% overlap
```

### **Test 8: Frame length = 10ms**
```python
frame_length_in_s: 0.010
frame_step_in_s: 0.005
```

### **Test 9: Frame length = 16ms**
```python
frame_length_in_s: 0.016
frame_step_in_s: 0.008
```

### **Test 10: Frame length = 32ms**
```python
frame_length_in_s: 0.032
frame_step_in_s: 0.016
```

### **Test 11: Frame length = 50ms**
```python
frame_length_in_s: 0.050
frame_step_in_s: 0.025
```
*(Mantieni il frame_length migliore)*

conf iniziale

---

### **Test 12: Overlap 0%**
```python
frame_step_in_s: frame_length_in_s  # usa il frame_length migliore trovato
```

### **Test 13: Overlap 25%**
```python
frame_step_in_s: frame_length_in_s * 0.75
```

### **Test 14: Overlap 75%**
```python
frame_step_in_s: frame_length_in_s * 0.25
```
*(Mantieni l'overlap migliore tra 0%, 25%, 50%, 75%)*

---

### **Test 15: n_mels = 20** (valore intermedio, se non già testato)
```python
n_mels: 20
n_mfcc: 20  # adatta anche questo
```

### **Test 16: n_mfcc = 20** (se n_mels >= 20)
```python
n_mfcc: 20
```

---

**Dopo ogni test:**
1. Confronta `test_accuracy` con il baseline
2. Se migliora, **mantieni** il valore cambiato
3. Se peggiora, **torna** al valore precedente
4. Procedi al test successivo

Il CSV salverà tutto automaticamente per l'analisi finale!

---

## **Esercizio 2: Magnitude-Based Weights Pruning**
**File:** `7.2 - Lab4 - Training with Weight Pruning.ipynb`

**Obiettivo:** Applicare pruning basato su magnitudine dei pesi

**Passi:**
1. Completare il notebook fornito
2. Configurare pruning: start=499, end=1499, amount=0.1, ogni 100 steps
3. Sperimentare con diversi valori di `prune_amount` e preprocessing
4. Valutare accuratezza e dimensioni ONNX (pre/post ZIP)
5. Deployare su Raspberry Pi e misurare latenza

---

## **Esercizio 3: Node-Based Pruning**
**File:** `7.3 - Lab4 - Training with Node-Based Pruning.ipynb`

**Obiettivo:** Applicare pruning basato su nodi con maschere

**Passi:**
1. Completare il notebook fornito
2. Sperimentare con diversi valori di regularization strength e preprocessing
3. Valutare accuratezza e dimensioni ONNX
4. Deployare su Raspberry Pi e misurare latenza

---

## **Esercizio 4: Post-Training Quantization**
**File:** `7.4 - Lab4 - Post-Training Quantization.ipynb`

**Obiettivo:** Quantizzare il modello CNN del Lab3

**Passi:**
1. Eseguire il notebook per applicare quantizzazione post-training
2. Valutare accuratezza e dimensioni del modello quantizzato
3. Deployare su Raspberry Pi e misurare latenza
4. **Analisi comparativa finale:** Confrontare le 4 tecniche e determinare:
   - Quale riduce meno l'accuratezza
   - Quale riduce più la memoria
   - Quale riduce più la latenza
   - Miglior trade-off per ottimizzare memoria/latenza minimizzando perdita di accuratezza