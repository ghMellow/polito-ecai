# 🎯 Testing Strategy per Raggiungere 99.4% Accuracy

## 📋 Ordine di Test Consigliato
Ti dico in chat quale ottimizzazione implementare e il codice relativo.

### 1️⃣ **STRATEGY 1: Quick Win (CONSIGLIATO INIZIARE QUI)**
**Tempo stimato:** 30-45 min  
**Probabilità successo:** ⭐⭐⭐⭐ (Alta)

**Cosa cambia:**
- ✅ Modifiche minime al codice esistente
- ✅ Label smoothing (riduce overfitting)
- ✅ AdamW optimizer (migliore generalizzazione)
- ✅ Learning rate più basso (0.0005 vs 0.001)
- ✅ Batch size ridotto (32 vs 64)
- ✅ Cosine annealing scheduler

**Setup:**
```python
# Sostituisci solo la sezione CFG e optimizer/scheduler
# Mantieni il modello KeywordSpotter originale
```

**Aspettative:**
- Test accuracy: 99.2-99.5%
- Size: ~2500 KB (poi quantizzare)

---

### 2️⃣ **STRATEGY 2: Model Enhancement**
**Tempo stimato:** 45-60 min  
**Probabilità successo:** ⭐⭐⭐⭐⭐ (Molto Alta)

**Cosa cambia:**
- ✅ Residual connections (migliore training)
- ✅ Attention mechanism
- ✅ Dual pooling (GAP + GMP)
- ✅ Model più profondo e robusto

**Setup:**
```python
# Usa KeywordSpotterV2 invece di KeywordSpotter
# Applica CFG_V2
```

**Aspettative:**
- Test accuracy: 99.4-99.7%
- Size: ~2800 KB (richiede quantizzazione più aggressiva)

---

### 3️⃣ **STRATEGY 3: Advanced Augmentation**
**Tempo stimato:** 60-75 min  
**Probabilità successo:** ⭐⭐⭐⭐ (Alta ma complessa)

**Cosa cambia:**
- ✅ MixUp data augmentation
- ✅ SpecAugment
- ✅ Volume variation
- ✅ Training function modificata

**Setup:**
```python
# Sostituisci AudioAugmentation con AdvancedAudioAugmentation
# Aggiungi SpecAugment
# Usa train_epoch_advanced
```

**Aspettative:**
- Test accuracy: 99.3-99.6%
- Size: simile a Strategy 1/2

---

## 🎲 Combinazioni Ottimali

### 🏆 **COMBO BEST (Strategy 1 + Strategy 2)**
**Raccomandazione TOP:**
```python
1. Usa KeywordSpotterV2 (modello migliorato)
2. Applica CFG da Strategy 1 (hyperparameters ottimizzati)
3. Mantieni augmentation semplice
```

**Perché:**
- Model più robusto + training ottimizzato
- Bilanciamento complexity/performance
- Più facile da debuggare

---

### ⚡ **COMBO AGGRESSIVE (Tutte e 3)**
Se hai tempo e vuoi massimizzare:
```python
1. KeywordSpotterV2
2. CFG_V3 hyperparameters
3. AdvancedAudioAugmentation + SpecAugment + MixUp
```

**Attenzione:** Training più lungo, possibile overfitting

---

## 🔧 Quick Fixes Aggiuntivi

### Seed Diversity
```python
# Prova seeds diversi - a volte fa la differenza
seeds = [42, 123, 777, 2024]
for seed in seeds:
    CFG['seed'] = seed
    # train...
```

### Test-Time Augmentation (TTA)
```python
def test_with_tta(model, feature_extractor, test_loader, device, n_tta=5):
    """Test con multiple augmentations e averaging"""
    model.eval()
    all_preds = []
    
    for _ in range(n_tta):
        correct = total = 0
        with torch.no_grad():
            for batch in test_loader:
                x = batch['x'].squeeze(1).to(device)
                y = batch['y'].to(device)
                
                # Light augmentation anche in test
                if _ > 0:  # Skip first pass (no aug)
                    x = light_augment(x)
                
                features = feature_extractor(x)
                output = model(features)
                predictions = output.argmax(dim=1)
                correct += (predictions == y).sum().item()
                total += y.size(0)
        
        all_preds.append(correct / total)
    
    return np.mean(all_preds) * 100
```

---

## 📈 Debugging Tips

### Se accuracy rimane < 99.4%:

1. **Check dataset balance**
```python
train_labels = [train_dataset[i]['y'].item() for i in range(len(train_dataset))]
print(f"Stop: {train_labels.count(0)}, Up: {train_labels.count(1)}")
```

2. **Analizza errori**
```python
def analyze_errors(model, feature_extractor, loader, device):
    model.eval()
    errors = {'stop_as_up': [], 'up_as_stop': []}
    
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            x = batch['x'].squeeze(1).to(device)
            y = batch['y'].to(device)
            features = feature_extractor(x)
            output = model(features)
            preds = output.argmax(dim=1)
            
            for i in range(len(y)):
                if preds[i] != y[i]:
                    if y[i] == 0:
                        errors['stop_as_up'].append(idx * len(y) + i)
                    else:
                        errors['up_as_stop'].append(idx * len(y) + i)
    
    print(f"Stop classified as Up: {len(errors['stop_as_up'])}")
    print(f"Up classified as Stop: {len(errors['up_as_stop'])}")
    return errors
```

3. **Ensemble di modelli**
```python
# Allena 3-5 modelli con seeds diversi
# Fai voting prediction
```

---

## ⏱️ Timeline Suggerita

**Giorno 1:**
- Mattina: Test Strategy 1 (quick win)
- Pomeriggio: Test Strategy 2 se Strategy 1 < 99.4%

**Giorno 2 (se necessario):**
- Mattina: Test Combo Best
- Pomeriggio: TTA e fine-tuning

---

## 🎯 Checklist Prima di Inviare

- [ ] Test accuracy ≥ 99.4%
- [ ] Verificato su multiple runs (almeno 3)
- [ ] Model size < 300 KB (dopo quantizzazione)
- [ ] Salvati ONNX models
- [ ] Testato inference su RPI (latency < 5ms)

---

## 💡 Note Finali

**Ricorda:** Sei già al 99%, quindi:
- Non servono rivoluzioni, solo ottimizzazioni
- La differenza può essere randomness - prova seeds diversi
- A volte meno augmentation è meglio
- Batch size più piccoli aiutano con pochi dati

**Good luck!** 🍀