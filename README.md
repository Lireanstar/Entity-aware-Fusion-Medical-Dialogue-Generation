# Entity-aware-Fusion-Medical-Dialogue-Generation (EFMDG)
This repository contains code and checkpoints for "Distinct but Correct: Generating Diversified and Entity-revised Medical Response".
#### The more details will be updated soon.
## 2021-10-20
### To add the code of the boost curriculum 5-fold, which is used for finetuning the EFMDG model. 
#### The designing of the 5-fold curriculum learning
1. The original pre-trained model (i.e., BertGPT-Entity) is utilized to initialize the parameters of the
encoder and decoder, which is fine-tuned with the cleaned online medical dialogues. Then, we use
the boost strategy to train 4 epochs for a total of 5-fold;
2. The dialogues with entities of the doctor is used for training, so that the generated response will
contain the common features of doctors. We use the boost strategy to train 4 epochs for a total of
5-fold;
3. We further sort out the dialogues with entities of doctors, whose length is greater than 11 (counted
on the validation set) to train the dialogue generation model, because these dialogues have more
entity characteristics. It is easier for the model to adapt to generating longer sentences. We train
2 epochs for a total of 5-fold.
#### The reason for designing the boost 5-fold
Although we used the validation set for training, we found that for the generation task, fitting the validation set can better enhance the generalization performance of the model. This method can help generate models to fit difficult samples (different splits) in curriculum learning. In this way, we can further obtain better results than the original BerTGPT model. The experimental results show that the strategy we proposed is effective. 

**Question**: Why to choose the boost training rather than directly fine-tuning the whole training set?

**Remarks**: The boost means to continuously train the model initialized with the last trained weight. We save the last trained model as the optimal model. To use the full data is easy to fall into local optimum. After the different splits (5-fold), different training sets are helpful for jumping out of the local best points.

