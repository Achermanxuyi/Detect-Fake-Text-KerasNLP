# LLM - Detect AI Generated Text

*Competition Link: https://www.kaggle.com/competitions/llm-detect-ai-generated-text*

Reference Code: https://www.kaggle.com/code/awsaf49/detect-fake-text-kerasnlp-tf-torch-jax-train

My version: https://www.kaggle.com/code/wanyingxu2004/detect-fake-text-kerasnlp-tf-torch-jax-infer/edit



This notebook demonstrate the usage of the multiple-backend capabilities of **KerasCore** and **KerasNLP** for the *Detect Fake Text*.

## Configuration

```python
class CFG:
    verbose = 0  # Verbosity
    device = 'GPU'  # Device
    seed = 42  # Random seed
    batch_size = 6  # Batch size
    drop_remainder = True  # Drop incomplete batches
    ckpt_dir = "/kaggle/input/daigt-kerasnlp-ckpt"  # Name of pretrained models
    sequence_length = 200  # Input sequence length
    class_names = ['real','fake']  # Class names [A, B, C, D, E]
    num_classes = len(class_names)  # Number of classes
    class_labels = list(range(num_classes))  # Class labels [0, 1, 2, 3, 4]
    label2name = dict(zip(class_labels, class_names))  # Label to class name mapping
    name2label = {v: k for k, v in label2name.items()}  # Class name to label mapping
```

