from langchain.text_splitter import RecursiveCharacterTextSplitter
# Sample long text (a paragraph from an article)
text = """ Quantum computing is an advanced computing paradigm that leverages quantum mechanics to perform computations at an unprecedented scale. 
          Unlike classical computers that use bits, quantum computers use qubits, which can exist in superposition states. 
          This enables quantum computers to solve problems that would take classical computers an impractical amount of time."""

# Initialize the text splitter with chunk overlap (Sliding Window)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,     # Define chunk size
    chunk_overlap=20   # Introduce overlap to retain context
)

# Split text into overlapping chunks					
chunks = text_splitter.split_text(text)

# Display the generated chunks
for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i+1} ---\n{chunk}")
