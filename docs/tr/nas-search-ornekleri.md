# NASearch Ornekleri

Bu dokuman, `pathologic.nas.NASearch` sinifini hizli sekilde kullanmak icin ornekler sunar.

## 0) Ergonomik API: `NASearch.for_model(...)`

Bu kullanimda `evaluate_candidate` fonksiyonunu manuel yazmaniz gerekmez.

```python
from pathologic.nas import NASearch

model_search = NASearch.for_model(
    "logreg",
    strategy="low_fidelity",
    random_state=42,
    max_evaluations=8,
)

result = model_search.search(
    search_space={
        "c": {"type": "float", "low": 0.1, "high": 3.0},
        "max_iter": {"type": "int", "low": 120, "high": 300, "step": 60},
    },
    x_train=x_train,
    y_train=y_train,
    x_val=x_val,
    y_val=y_val,
    n_candidates=16,
    budget={"min_fidelity": 1, "max_fidelity": 3},
)

print(result.best_candidate.params)
print(result.best_score)
```

Not: Low-fidelity stratejisi adaylara varsayilan olarak `epochs` parametresi ekler.
`NASearch.for_model(...)` bu parametreyi varsayilan olarak model constructor'a iletmez
(`fidelity_param_key="epochs"`). Eger `epochs` parametresini modele gecirmek isterseniz
`fidelity_param_key=None` verebilirsiniz.

## 1) En Basit Kullanim (Low-Fidelity)

```python
from pathologic.nas import NASearch

search = NASearch(
    strategy="low_fidelity",
    random_state=42,
    max_evaluations=12,
    direction="maximize",
)

search_space = {
    "c": {"type": "float", "low": 0.1, "high": 3.0},
    "max_iter": {"type": "int", "low": 120, "high": 300, "step": 60},
}


def evaluate_candidate(params: dict[str, float]) -> float:
    # Burada kendi objective fonksiyonunuzu calistirin.
    # Ornek olarak daha yuksek c degerini odullendirelim.
    return float(params["c"])

result = search.search(
    search_space=search_space,
    evaluate_candidate=evaluate_candidate,
    n_candidates=20,
    budget={"min_fidelity": 1, "max_fidelity": 4},
)

print(result.best_candidate.params)
print(result.best_score)
print(result.stopped_reason)
```

## 2) Weight-Sharing Stratejisi

```python
from pathologic.nas import NASearch

search = NASearch(
    strategy="weight_sharing",
    random_state=7,
    max_evaluations=16,
    strategy_kwargs={
        "shared_keys": ["backbone_width"],
        "shared_groups": 2,
    },
)

search_space = {
    "backbone_width": {"type": "categorical", "values": [32, 64, 128]},
    "dropout": {"type": "float", "low": 0.0, "high": 0.4},
}


def evaluate_candidate(params: dict[str, float]) -> float:
    # Kendi model egitiminiz + validasyon skorunuz burada hesaplanir.
    return 1.0 - float(params["dropout"])

result = search.search(
    search_space=search_space,
    evaluate_candidate=evaluate_candidate,
    n_candidates=24,
    budget={"min_fidelity": 1, "max_fidelity": 3, "shared_groups": 2},
)

print(result.best_candidate.metadata)
```

## 3) Erken Durdurma (Patience)

```python
from pathologic.nas import NASearch

search = NASearch(
    strategy="low_fidelity",
    random_state=42,
    patience=3,
    min_improvement=0.0,
    max_evaluations=100,
)

search_space = {
    "x": {"type": "float", "low": 0.0, "high": 1.0},
}

result = search.search(
    search_space=search_space,
    n_candidates=50,
    evaluate_candidate=lambda params: 1.0,  # hic iyilesme yok
    budget={"min_fidelity": 1, "max_fidelity": 2},
)

print(result.stopped_reason)  # genelde early_stopping
print(len(result.trials))
```

## 4) Sure Butcesi ve Trial Butcesi

```python
from pathologic.nas import NASearch

search = NASearch(
    strategy="low_fidelity",
    random_state=42,
    max_evaluations=10,
    max_seconds=5.0,
)

search_space = {
    "lr": {"type": "float", "low": 0.0005, "high": 0.01},
}

result = search.search(
    search_space=search_space,
    n_candidates=100,
    evaluate_candidate=lambda params: -abs(float(params["lr"]) - 0.003),
    budget={"min_fidelity": 1, "max_fidelity": 5},
)

print(result.stopped_reason)  # completed | max_evaluations | timeout | early_stopping
```

## 5) Secilen Adayi Model Egitimine Tasima

Asagidaki ornek, NAS sonucu ile en iyi parametreleri alip model olusturmayi gosterir.

```python
from pathologic.models import create_model
from pathologic.nas import NASearch

search = NASearch(strategy="low_fidelity", random_state=42, max_evaluations=8)

search_space = {
    "c": {"type": "float", "low": 0.1, "high": 3.0},
    "max_iter": {"type": "int", "low": 120, "high": 300, "step": 60},
}

result = search.search(
    search_space=search_space,
    n_candidates=16,
    evaluate_candidate=lambda p: float(p["c"]),
    budget={"min_fidelity": 1, "max_fidelity": 3},
)

best = result.best_candidate.params
model = create_model(
    "logreg",
    random_state=42,
    model_params={
        "c": float(best["c"]),
        "max_iter": int(best["max_iter"]),
    },
)
```

## Notlar

- Reproducibility icin `random_state` sabit verin.
- `search_space` parametre tipleri: `float`, `int`, `categorical`.
- `budget` sozlugu stratejiye gore farkli alanlar alabilir:
  - low_fidelity: `min_fidelity`, `max_fidelity`
  - weight_sharing: `min_fidelity`, `max_fidelity`, `shared_groups`, `shared_keys`
- Sonuc objesi `best_candidate`, `best_score`, `trials`, `stopped_reason` alanlarini icerir.
