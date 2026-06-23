"""
- Compare the selected ALGORITHMS in the different scenarios, testing the actual results (curriculum learning methodology)
 [Test the actions executed during each epoch ak, to retrieve the real improvement)
- Training plots for both the models with the selected ALGORITHMS (entropy, avg num of steps, success rate, total return, joint episodic rewards).
-Telemetry Pipeline Bottlenecks: Assess the computational overhead of parsing the Snort alert logs (/var/log/snort/alerts) during the step transition.
 Show that populating the Blue Team's observation vector does not critically bottleneck the training throughput. (EXTRA)

- Comparison the execution phase between the ATP custom and the baseline in both cyber defense scenarios (metrics: success rate, total return, avg num of steps)
//the blue agent converts into an element of the environment, like an hidden entity)
- Testing the agent trained in scenario Tiny on the Medium one, and vice versa (metrics: success rate, total return, avg num of steps) (MOCK?)
 //Generalization capabilities
- Exploitability score: Analyze the diversity of the generated attack vectors. Does the agent rely solely on Samba exploits, or does it dynamically pivot to 
SSH or HTTP based on the firewall topology?
Cross-Play (Heuristic Payoff Matrix): Extract policy checkpoints at regular intervals during training (e.g., Epoch 100, 500, 1000). 
Run an evaluation tournament where historical Red policies are pitted against historical Blue policies. A robust co-evolutionary setup will show that later 
generations of the Blue Team consistently defeat early generations of the Red Team, proving monotonic learning progression.
"""