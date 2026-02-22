# MNIST MLPS

Run a baseline experiment with the most normal results with zero calibration and regular training (20 epochs). 

Experiment 1:

Train a distillation experiment in which the teacher is only trained for 5, 10, 15, 20 epochs. Then, train a smaller model on this for distillation and measure its accuracy. Do 3 seeds per test.
* This experiment will tell us how the accuracy of teh teacher (increasing with epochs) will affect the accuracy of the student.
    * If the accuracy of the student is increasing with the teacher, we are awawer that the teacher's value comes from accuracy.

Train a distillation experiment in which the teacher is increasing in size (hopefully better at generalizing) and then train a smaller model on this for distillation and measure its accuracy. Do 3 seeds per test. 
* This experiment will tell us how the value fo the teacher comes from model size, which can be correlated wtih accuracy or not and is useful to know. 
    * If the accuracy of the student increase swith param of teacher (and is more accurate) we see value in accuracy
    * Otherwise we see that the model is overfitting and that the accuracy may not be a factor.


## calibration

Train a distillation experiment using temperature scaling where the model's logits are divided by an increasing T. Do 3 seeds per test. 
* The idea is that as T increases, the model is less confident. Measure the expected calibration error of this teacher model, and also measure the corresponding performance of the student model. 

Train a distillation experiment with tradeoffs between T values and epochs (accuracy). Since T should not have an effect on accuracy we should try and measure the discrepancies between the model performance as accuracy increases, T decreases. 
* This experiment will tell us whether these variables are confounding if at all and which combinations yield the best performance. Of course, run these in seed 3 experiments. 

* It might also help to run ablations where the model is completely inaccurate and then do temperature adjustemnet to evaluate the performance. 
Then we can evaluate the difference from the original baseline with a p value test? 


# Question 2

Task: What happens in a distribution shift? 

Experiment 1:
* Train a teacher on all 10 mnist classes, and train a student on all 10 mnist classes
* Then evaluate a student on 0-5 classes.
* This experiment will tell us whether the teacher's konwledge will transfer to classes that the student has not seen .
    * If the classes are transferable, it tell us that distillation can handle distribution mismatch. 
    * if not, then it tells us that it cannot handle distribution mismatch. 

-> In the case that it does hanlde distribution mismatch, it would be worthwhile to run tests that determine whether occlusion (a differnet type of distribution mismatch rather than just leaving ot whole digits) would be yielding similar or differnet results. 
The experiment is as follows:
* Train the teacher model on full images and then train the student model on levels of occlusion (From 0 to increasing 100 percent). 
    * If we see that performance is still quite high even with high occlusion then the labels are quite informative. 
    * Otherwise we can pay attention to what occclusion levels perform best.

* In this case, it would also be worthwhile to run experiments that vary the Temperature value, and then we can train the models on whether that affects anything in the poor performing cases. (Of course this is a side experiment only if extra time)

* The final experimetn would be the reverse, where the teacher is trained on a few and the student is trained on all 10. 
    * If the reverse direction fails then we see that the stronger model must have more training data than the student model and the studnet model is directly dependent on the teacher model to obtain its values.
    * Otherwise we see that the student model maintains its autonomy.

A good follow up to this would be to alter alpha and see how much the studnet model relies on the teacher and student (when the teacher is trained with less data) 
* If the alpha increases with student student accuracy (meaning more student reliance) we see that the model is targeted toward more of the harder labels when the teacher model has less knowledge.
* Then if the alpha decreases with student accuracy then we see that the soft labels of the teacher models still hold importance. 