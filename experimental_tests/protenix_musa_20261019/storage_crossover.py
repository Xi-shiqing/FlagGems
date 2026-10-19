"""Four complete-model variants, six alternating blocks, identical input."""
import crossover_benchmark as cross
from attention_compat import wrap
from bias_softmax import fused_attention
from bias_softmax_storage import separate,hybrid

cross.variants={'native':cross.native_attention,'inplace':wrap(fused_attention),
                'separate':wrap(separate),'hybrid':wrap(hybrid)}
cross.order=['native','inplace','separate','hybrid',
             'hybrid','separate','inplace','native',
             'inplace','separate','hybrid','native',
             'native','hybrid','separate','inplace',
             'separate','hybrid','native','inplace',
             'inplace','native','hybrid','separate']
cross.experiment_description='Four storage variants, six alternating four-call blocks; all generation settings unchanged'
cross.shared.inference.InferenceRunner.predict=cross.crossover
cross.shared.inference.run()
