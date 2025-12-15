import torch
import torch.nn as nn
from model import SlotTransformer
from data import generate_family_batch

NUM_PEOPLE = 5
NUM_PROPS = 2 * (NUM_PEOPLE ** 2) # Total slots (Parent + Ancestor)
EMBED_DIM = 64
BATCH_SIZE = 32
LR = 0.001

# Initialize Model
model = SlotTransformer(
    num_propositions=NUM_PROPS,
    embed_dim=EMBED_DIM,
    num_heads=4,
    num_layers=2
)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCELoss() # Binary Cross Entropy (Standard for True/False)

print("Starting Training...")

for epoch in range(1000):
    # 1. Get Data
    # inputs: We know parents, but Ancestor slots are 0
    # targets: We know parents AND inferred Ancestors
    inputs, targets = generate_family_batch(BATCH_SIZE, num_people=NUM_PEOPLE)
    
    # 2. Forward Pass
    # model guesses truth scores for all propositions
    predicted_scores = model(inputs)
    
    # 3. Compute Loss
    # We want the model's output to match the target logic closure
    loss = criterion(predicted_scores, targets)
    
    # 4. Backward Pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
        
# -- Test it --
print("\nTesting Inference:")
test_input, test_target = generate_family_batch(1, num_people=NUM_PEOPLE)

# Let's say A->B and B->C
# We manually inject this into slot 0->1 and 1->2 (just conceptually)
# The model should output high probability for Ancestor(A,C)
with torch.no_grad():
    prediction = model(test_input)
    
# Check one "Ancestor" slot that should be true
# (We filter for slots where input was 0 but target is 1)
mask_hidden_truths = (test_input == 0) & (test_target == 1)
recovered_truths = prediction[mask_hidden_truths]

print(f"Average confidence on inferred truths: {recovered_truths.mean().item():.2f}")
# If this is close to 1.0, the model learned the logic!