# Linear Algebra

Linear algebra is the branch of mathematics dealing with continuous data (rather than discrete integers). It is the primary mathematical language used ot describe and manipulate data in ML algorithms.

## 1. The Building Blocks

These are the four main types of "objects" we work with.

- Scalars: A single number (e.g., $s=5$). In deep learning, this might represent a single parameter like a learning rate.
- Vectors: An array of numbers arranged in order (e.g., $x = [1,2,3]$). We cna think of a vector as a point in space, where each number is a coordinate along a different axis.
- Matrices: A 2-D array of numbers (a grid with rows and columns). A matrix often represents a dataset (where rows are examples and columns are features) or a linear transformation.
- Tensors: An array with more than two axes. For example, and RGB image is a 3-D tensor (Height x Width x Color Channels).

## 2. Basic Operations

How these objects interact with each other.

- Transpose ($A^T$): Flipping a matrix over its main diagonal. The rows become columns, and the columns become rows. Put another way, the examples become features and the features become examples.
- Broadcasting: A rule that allows you to add a smaller object (like a vector) to a larger object (like a matrix). For example, if you add a vector $b$ to a matrix $A$, the vector is implicitly copied to every row of the matrix so the addition can happen.
- Matrix Multiplication: The standard way to multiply matrices. You take the dot product of the rows of the first matrix with the columns of the second.
- Hadamard Products: This is element-by-element multiplication.
- Dot Product: Multiplying two vectors to get a single scalar. It tells you how much two vectors point in the same direction.

## 3. Measuring Size (Norms)

In ML, we often need to measure how "large" a vector is or how far apart two vectors are (error). We use functions called Norms ($L^p$).

- $L^2$ Norm (Euclidean Norm): The standard distance from the origin. It is the square root of the sum of squared elements.
- $L^1$ Norm: The sum of the absolute values of the elements. It is often used when we want a solution that is "sparse" (has many zeros).
- Max Norm ($L^\infty$): The absolute value of the largest elements in the vector.
- Frobenius Norm: Used to measure the size of a matrix (similar to the $L^2$ norm for vectors).

## 4. Matrix Properties & Decompositions

Just as we can factor the number $12$ into $2 \times 2 \times 3$ to understand it better, we can break matrices down to understand their properties.

### Identity and Inverse

- Identity Matrix ($I$): A special square matrix with 1s on the diagonal and 0s everywhere else. Multiplying by it changes nothing ($AI = A$), similar to multiplying a number by 1.
- Inverse Matrix ($A^{-1}$): A matrix that "undoes" the work of $A$. If $Ax=b$, then $x=A^{-1}b$.
    - Note: Not all matrices have an inverse. If a matrix effectively "collapses" space (making two different inputs produce the same output), it cannot be inverted.

### Eigendecomposition

This breaks a square matrix into a set of eigenvectors and eigenvalues.

- Eigenvector: A special vector that doesn't change direction when the matrix transforms it; it only stretches or shrinks.
- Eigenvalue: The number that indicates how much the eigenvector stretches (or shrinks).
- *Application:* This helps us understand what a matrix does to space (e.g., stretching it along specific axes).

### Singular Value Decomposition (SVD)

A more general version of Eigendecomposition that works for any matrix (even non-square ones). It breaks a matrix into three parts $U$, $D$, and $V$. This is crucial for tasks like data compression and denoising.

## 5. Advanced Concepts

- Pseudoinverse (Moore-Penrose): Since not all matrices have an inverse (especiall non-square ones), we use the pseudoinverse to find a "best fit" solution. It is used to solve linear equations where there is no perfect solution or infinitely many solutions.
- Trace: The sum of all the elements on the main diagonal of a matrix.
- Determinant: A single number calculated from a square matrix that tells you how much the matrix expands or contracts volume.
    - If the determinant is 0, the matrix collapses space completely (and has no inverse).
    - If the determinant is 1, the matrix preserves volume.

## 6. Example: Principle Component Analysis (PCA)

The chapter concludes by showing how these concepts combine to build PCA, a simple machine learning algorithm. PCA uses linear algebra to compress data by finding a lower-dimensional representation (encoding) that loses as little information as possible. It does this by finding the directions (eigenvectors) where the data varies the most.

---

In the daily life of a researcher working on Transformers, linear algebra provides the intuition for architecture design, debugging code, and optimizing performance.

### 1. Tensors are the Data Structure

You rarely think in "scalars" or "vectors". You think almost exclusively in Tensors.

- Day-to-day: Your data is usually a 4D tensor: `[Batch Size, Sequence Lenght, Height, Width]` for images, or a 3D tensor: `[Batch Size, Context Window, Embedding Dimension]` for text.
- The struggle: 50% of debugging is fixing "Shape Mismatch" errors. You constantly manipulate these tensors (transposing, reshaping, squeezing) to ensure the math flows correctly.

### 2. Matrix Multiplication is the Engine

The "Linear Layer" (or Dense Layer) is the workhorse of neural networks.

- The Mechanism: In Transformers, projecting inputs into Queries, Keys, and Values ($Q, K, V$) is just three massive matrix multiplications.
- Dot-Products = Similarity: In Self-Attention, you calculate the dot product between a Query vector and a Key vector. Pragmatically, this measures relevance. A high dot product means the model should "pay attention" to that specific token.

### 3. Broadcasting is Key to Attention

Broadcasting (applying smaller arrays to larger ones) is essential for efficiency.

- Pragmatic use: When implementing Masked Attention (so the model can't see the future tokens), you create a simple mask matrix of $-\infty$ and "broadcast" add it to the huge attention score matrix. This forces the probabilities of future tokens to zero without needing a loop.

### 4. Norms Ensure Stability

Deep networks are notoriously unstable (gradients explode or vanish).

- L2 Norm (Euclidean): Used in Gradient Clipping. If the gradient vector is "too long" (too large a magnitude), you scale it down based on its L2 norm to prevent the model from diverging.
- Layer Normalization (RMSNorm): Modern LLMs use variants of normalization (forcing activations to have specific mean/variance). This is purely checking vector statistics to ensure the model learns smoothly.

### 5. Decomposition (SVD) & Low-Rank Adaptation (LoRA)

This is currently the hottest topic in Generative AI fine-tuning.

- The Concept: SVD teaches us that large matrices can be approximated by multiplying smaller matrices.
- Pragmatic use: Instead of retraining a massive 70B parameter model, researchers use LoRA. They freeze the big weights and train two tiny matrices ($A$ and $B$) whose product approximates the update. This relies on the linear algebra concept that the "update matrix" is Low Rank (it has little new information compared to its size).

### 6. Sparse Matrices (Efficiency)

- The Concept: Matrices populated mostly with zeros.
- Pragmatic use: In "Mixture of Experts" (MoE) models, you only activate a small subset of the neural network for each token. Mathematically, this is treated as multiplying by sparse matrices to save massive amounts of compute.
