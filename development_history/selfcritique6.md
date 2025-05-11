Strengths:

- Transformer model is outputting the outer-most nesting correctly 96% of the time
- Regex parsing for sympy is complete and validated, ready for use after transformer.

Areas for Improvement:

- Force model to output more than the outer-most operation nesting (currently outputting only add, subtract, multiply, divide)

Critical Risks:

- Unsure if text-to-text transformations are going to be more difficult for the transformer or require more data.

Concrete Next Actions:

- First priority is continuing the model training and fixing the limited output issue.
  
Resources Needed:

- None currently. Pending more data later if needed.
