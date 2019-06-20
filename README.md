# pre_maml
some pre-test of applying "maml" on new task


## Trouble Shooting
1. f-string used in openai spinningup only supported by py3.6.  
   ```
   pip install future-fstrings
   ```
   and at the beginning of code file, add
   ```
   # -*- coding: future_fstrings -*-
   ```
