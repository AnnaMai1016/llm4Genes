# Extraction assessment

- Before going to use the model to perform extraction. I plan to assess the performance of the extraction with difference settings across 50 random sampled papers. Here are several considerations:
  - Using models with different intellegent level: 
    - `GPT-5.4-mini`
    - `GPT-5-mini`
  - Using different extraction strategies:
    - `Extraction only: Single path extraction`. In which all field are extraction in one turn. 
    - `Extraction only: Multi path extraction`. For instance, five path extractions:

      ```python
      NAME_FIELDS = [
          ['metadata', 'relevance'],
          ['context'],
          ['scale','methodology','tile_design'],
          ['results','internal_evaluation',],
          ['key_words','overall_synthesis',]
      ]
      ```

    - `Extraction+Update`: Single path extraction + 1 and 2 turn update.
    - `Extraction+Verification`: Single path extraction + 1 turn verification and pathes apply.
    - `Extraction+Update+Verification`: Single path extraction + 1 turn update + 1 turn verification and path apply.

To reduce the cost, we should use *batch api* and update the results offline. Besides, `GPT-5.4-mini` only perform `Extraction only: Single path extraction`. For other cases, the extraction can be done sequentially, for example:

- For `Extraction+Update`: it can read the extraction results from `Extraction only: Single path extraction`, and perform update. Also, 2 turn update should based on the results from 2 turn update.
- For `Extraction+Verification`: it can read the extraction results from `Extraction only: Single path extraction`, and perform verifiction. Then apply batches to get the final results.
- For `Extraction+Update+Verification`: it can read the 1 turn updated results from `Extraction+Update`, and perform verifiction. Then apply batches to get the final results.