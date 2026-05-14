add another perturbation of not closing the gripper enough, like, close a little bit but still reaching the target 

add one where it closes its gripper too early, even though its not at the target yet and then hovers around there, like underreach idle but with an early close

vertical drift?

reaches the right area but doesn't close the grippers all the way, like 50% of it and then sits there, or like it fluncuates gripper from like 0.3 to 0.6 or something but not enough to fully hold whatever it's picking up.

semantic perturbations like you have two trajectories, one for red and one for blue, but you swap the target and show the red trajectory with a blue target and vice versa, so red picks up blue and blue picks up red and then you can label it as like "wrong target, meant to grab red" or "wrong object, meant to pick up red cube" etc.


Use this as like a post training strategy to distill recovery

Train a model that is basically added to whatever we are currently generating actions with, like a error correction policy and it just learns to predict the residual of the action that was supposed to be taken.




Interesting explanation of goal.

I want to make sure you understand the goal because I'm not sure if you get it yet. So focusing just on the a2l-pr folder, we made a framework/system for perturbating successful trajectories to try to make them into failing one (right now we are using robomimic square dataset but I want to use this for other datasets too (specifically real world agilex piper arm data, not sure how I'll deal with that though if we end up also relying on image data since we aren't synthetically creating failing images with the failing trajectories; not a problem for robomimic since we can just simulate)). Then, the idea is train some kind of classifier, language model, etc. that learns to detect potential errors during policy inference. We currently only have 4 perturbations to start with but hoping to expand. The goal is to embed some kind of primitive recovery into our robot system. each synthetic perturbation also records recovery 'instructions' of sorts--these may need to be modified--but the hope is to design a robust model that recognize when a trajectory is messing up, like it is underreaching and is hovering just before whatever it is trying to pick up and just needs to go forward a little more, or it closed the gripper too early so when it goes to pick it up it detected the gripper needs to open and then the recovery opens it. Does this make sense? Do you have any questions? With this context now, can you help figure out why in the a2l-pr folder, specifically in #file:robomimic_full_training.ipynb 

Our current model seems to greatly overfit.
Epoch 15 | Train Loss: 0.3560 | Val Loss: 7.0990 | Val Acc: 16.50%

Can you help try to mitigate this. As well, we want to make sure our perturbations are long enough or catastrophic enough where the classifier actually has time to be able to detect an abnormality. Please really think about this one. I want it to be able to learn that oh its been there for a while trying to grasp at the object but it's just a little too far. Can you make sure everything makes sense for this kind of challenge. What kind of model is needed? How can we fix our current one? Is there a better approach? This has to be a situation that a model can realistically detect. Are we basing it on simulated image inputs too?