================================================================================
 RANDOM FOREST HYPERPARAMETER (ccp_alpha) TUNING FOR REAL-TIME ROBUSTNESS
================================================================================

In Random Forest, 'ccp_alpha' stands for Cost-Complexity Pruning Alpha.
During real-time prediction, deep, unpruned trees (alpha=0.0) can sometimes overfit
to specific noisy waveforms seen during training, causing false alarms later.
Increasing 'alpha' prunes the weakest branches of the trees, creating a simpler,
more robust, and faster model—at the potential cost of some training accuracy.

--------------------------------------------------------------------------------
Alpha      | Accuracy (%)    | Macro F1     | Training Time (s)  | Tree Depth (Avg)
--------------------------------------------------------------------------------
0.0000     | 100.0000        | 1.0000       | 6.95               | 20.4           
0.0001     | 100.0000        | 1.0000       | 6.74               | 20.0           
0.0005     | 100.0000        | 1.0000       | 6.76               | 19.2           
0.0010     | 100.0000        | 1.0000       | 6.87               | 18.5           
0.0050     | 93.1218         | 0.9294       | 5.11               | 13.9           
0.0100     | 69.8240         | 0.5394       | 5.46               | 5.8            
0.0500     | 61.0352         | 0.3821       | 5.48               | 0.9            
0.1000     | 54.3307         | 0.2347       | 5.07               | 0.0            
--------------------------------------------------------------------------------

================================================================================
 SELECTION GIVEN REAL-TIME CONSIDERATIONS
================================================================================
The model with ccp_alpha = 0.0 achieved the highest test accuracy of 100.0000%.

Analysis of Pruning (Alpha):
1. Alpha = 0.0 (No Pruning): The model builds extremely deep trees. While accuracy is high,
   it is highly complex and more likely to overfit strictly to training noise.
2. As Alpha increases (e.g., 0.001 - 0.01): The trees become shallower (average depth drops).
   The model becomes generalized, meaning it focuses only on the most dominant Electrical signatures (like major THD shifts).
   This makes it physically faster to compute in real-time embedded systems (microcontrollers).
3. Alpha >= 0.05: The pruning is too aggressive. The model underfits and accuracy drops significantly.

RECOMMENDATION FOR REAL-TIME DEPLOYMENT:
To prevent overfitting in a real-time environment, we recommend setting ccp_alpha = 0.001.
This applies minor pruning to remove edge-case branches, ensuring the model relies only on the most impactful, highly confirmed sensor thresholds. This avoids false positives and executes slightly faster on hardware.
