# Data

Place raw CSV files here if you use them.

Alternatively, load the Wisconsin breast cancer dataset in a notebook:

```python
from sklearn.datasets import load_breast_cancer
import pandas as pd

data = load_breast_cancer(as_frame=True)
df = data.frame
```

Do not commit large or sensitive files unless your course policy allows it.
