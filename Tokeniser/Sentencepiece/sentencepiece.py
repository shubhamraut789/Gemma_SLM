import sentencepiece as spm
import os
import gc

def train_large_corpus(input_file, model_prefix="multi_lang_spm", vocab_size=32000):
   
    print(f" Memory-Optimized SentencePiece Training")
    print(f" Directory: {os.getcwd()}")
    print(f" Input file: {input_file}")
    print(f" File size: {os.path.getsize(input_file):,} bytes")
    
    abs_input = os.path.abspath(input_file)
    abs_model_prefix = os.path.abspath(model_prefix)
    
    print(f" Output prefix: {abs_model_prefix}")
    
    training_params = {
        'input': abs_input,
        'model_prefix': abs_model_prefix,
        'vocab_size': vocab_size,
        'character_coverage': 0.9995,
        'model_type': 'unigram',
        'input_sentence_size': 200000,        # Reduced from 1M to 200K
        'shuffle_input_sentence': True,
        'max_sentence_length': 5000,          # Reduced from 10K to 5K
        'num_threads': 8,                     # Limit threads to reduce memory
        'max_sentencepiece_length': 16,
        'split_by_unicode_script': True,
        'split_by_whitespace': True,
        'split_by_number': True,
        'shrinking_factor': 0.75,
        'num_sub_iterations': 2
    }
    
    print(f" Training parameters:")
    for key, value in training_params.items():
        print(f"   {key}: {value}")
    
    try:
        print(f"\n Starting training with reduced memory footprint...")
        
        gc.collect()
        
        spm.SentencePieceTrainer.train(**training_params)
        
        print(f" Training completed!")
        
        model_file = f"{abs_model_prefix}.model"
        vocab_file = f"{abs_model_prefix}.vocab"
        
        success = True
        
        if os.path.exists(model_file):
            model_size = os.path.getsize(model_file)
            print(f" Model file: {model_file}")
            print(f"   Size: {model_size:,} bytes")
        else:
            print(f" Model file not found: {model_file}")
            success = False
        
        if os.path.exists(vocab_file):
            vocab_size_bytes = os.path.getsize(vocab_file)
            print(f" Vocab file: {vocab_file}")
            print(f"   Size: {vocab_size_bytes:,} bytes")
        else:
            print(f" Vocab file not found: {vocab_file}")
            success = False
        
        if success:
            # Test the model
            print(f"\n Testing model...")
            try:
                sp = spm.SentencePieceProcessor()
                sp.load(model_file)
                vocab_size_actual = sp.get_piece_size()
                print(f" Model loads successfully!")
                print(f"   Actual vocab size: {vocab_size_actual}")
                
                # Test encoding
                test_sentences = [
                    "Hello world!",
                    "This is a test sentence.",
                    "Machine learning is amazing."
                ]
                
                for test_text in test_sentences:
                    encoded = sp.encode_as_pieces(test_text)
                    decoded = sp.decode_pieces(encoded)
                    print(f"   Test: '{test_text}' → {len(encoded)} pieces")
                    print(f"         {encoded[:5]}..." if len(encoded) > 5 else f"         {encoded}")
                
                return True
                
            except Exception as e:
                print(f" Model test failed: {str(e)}")
                return False
        
        return success
        
    except Exception as e:
        print(f" Training failed: {str(e)}")
        
        print(f"\n Trying with ultra-conservative settings...")
        try:
            conservative_params = training_params.copy()
            conservative_params.update({
                'input_sentence_size': 50000,    # Even smaller sample
                'max_sentence_length': 2000,     # Much shorter sentences
                'num_threads': 4,                # Fewer threads
                'vocab_size': min(vocab_size, 16000)  # Smaller vocab if requested was large
            })
            
            print(f" Conservative parameters:")
            for key, value in conservative_params.items():
                if key in ['input_sentence_size', 'max_sentence_length', 'num_threads', 'vocab_size']:
                    print(f"   {key}: {value}")
            
            # Force garbage collection
            gc.collect()
            
            spm.SentencePieceTrainer.train(**conservative_params)
            
            # Check files again
            model_file = f"{abs_model_prefix}.model"
            vocab_file = f"{abs_model_prefix}.vocab"
            
            if os.path.exists(model_file) and os.path.exists(vocab_file):
                print(f" SUCCESS with conservative settings!")
                print(f"   Model: {os.path.getsize(model_file):,} bytes")
                print(f"   Vocab: {os.path.getsize(vocab_file):,} bytes")
                return True
            else:
                print(f" Conservative approach also failed")
                return False
                
        except Exception as e2:
            print(f" Conservative training also failed: {str(e2)}")
            return False

def cleanup_temp_files():
    temp_files = ['tiny_model.model', 'tiny_model.vocab', 'tiny_test.txt']
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                print(f" Cleaned up: {temp_file}")
            except:
                pass

if __name__ == "__main__":
    cleanup_temp_files()
    
    input_file = "final_train.txt"
    model_prefix = "multi_lang_spm"
    vocab_size = 32000
    
    print("=" * 70)
    print(" MEMORY-OPTIMIZED SENTENCEPIECE TRAINING")
    print("=" * 70)
    
    if not os.path.exists(input_file):
        print(f" Input file '{input_file}' not found!")
        exit(1)
    
    success = train_large_corpus(input_file, model_prefix, vocab_size)
    
    if success:
        print(f"\n SUCCESS!!")
        print(f" Files created:")
        print(f"   • {model_prefix}.model")
        print(f"   • {model_prefix}.vocab")
    else:
        print(f"\n FAILED!")
    
    print("=" * 70)
