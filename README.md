# nlp100-chapter10
言語処理100本ノック　第10章　事前学習済み言語モデル（GPT型）の実装
# SST-2 Prompt Optimization with Textual Feedback

SST-2の感情分類を対象に、LLMの誤分類や理由説明に対する評価結果を「言語的な勾配」として用い、プロンプトを反復的に改善する実験を行いました。

TextGradの考え方を参考に、推論 → LLM-as-a-Judgeによる評価 → 誤り傾向の集約 → プロンプト更新、という流れを実装しています。

最適化用198件では、初期プロンプトの分類精度86.9%から最良プロンプトで93.9%まで向上しました。
※同一データを最適化と評価に使用したin-sample評価です。

## Main Files

- `optimize_98_textgrad_rationale_root_separated_prompt.py`  
  評価結果から言語的なフィードバックを生成し、プロンプトを反復改善するメインコード。

- `evaluate_98_sst2_dev_root_best.py`  
  作成したプロンプトをSST-2データで評価するコード。

- `optimize_98_textgrad_prompt_llama_root_only.py`  
  理由説明を導入する前の、分類プロンプト最適化のベースライン実験。

- `result/.../prompt_metrics.jsonl`  
  各プロンプトの分類精度や評価結果を記録。

- `result/.../textual_gradients.jsonl`  
  誤り分析から生成した言語的な改善指針を記録。

## Technologies

Python / Llama 3.2 1B / LLM-as-a-Judge / TextGrad-inspired Prompt Optimization