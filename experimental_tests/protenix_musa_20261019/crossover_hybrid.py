import crossover_benchmark as cross
from attention_compat import wrap
from bias_softmax_storage import hybrid
cross.variants={'native':cross.native_attention,'fused':wrap(hybrid)}
cross.experiment_description='ABBA/BAAB: native vs precision-aware hybrid fusion, 8 calls each'
cross.shared.inference.InferenceRunner.predict=cross.crossover
cross.shared.inference.run()
