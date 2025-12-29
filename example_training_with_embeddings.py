"""
Validation script to test the sentence embedding alignment changes.
This script validates:
1. Dataloader can load embeddings correctly
2. GLIM model can process sentence embeddings
3. Backward compatibility is maintained
"""

import torch
import numpy as np
import pickle
import os
import sys

data_path = '/nfs/usrhome2/yguoco/checkpoints_sentiment_cls_with_mlp/tmp'

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_datamodule_without_embeddings():
    """Test that datamodule works without embeddings (backward compatibility)."""
    print("\n" + "="*80)
    print("Test 1: Datamodule without embeddings (backward compatibility)")
    print("="*80)
    
    try:
        from data.datamodule import GLIMDataModule
        
        # This should work even if embeddings_path is None
        datamodule = GLIMDataModule(
            data_path=data_path + '/zuco_merged.df',
            embeddings_path=None,
            bsz_train=2,
            bsz_val=2,
            bsz_test=2,
            num_workers=0
        )
        print("✓ GLIMDataModule initialized successfully without embeddings")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

def test_datamodule_with_embeddings():
    """Test that datamodule can load embeddings."""
    print("\n" + "="*80)
    print("Test 2: Datamodule with embeddings")
    print("="*80)
    
    # Check if embeddings file exists
    embeddings_path = data_path + '/embeddings.pickle'
    if not os.path.exists(embeddings_path):
        print(f"⚠ Embeddings file not found at {embeddings_path}")
        print("  This is expected if preprocessing hasn't been run yet.")
        return None
    
    try:
        from data.datamodule import GLIMDataModule
        
        # Load with embeddings
        datamodule = GLIMDataModule(
            data_path=data_path + '/zuco_merged.df',
            embeddings_path=embeddings_path,
            bsz_train=2,
            bsz_val=2,
            bsz_test=2,
            num_workers=0
        )
        print("✓ GLIMDataModule initialized successfully with embeddings")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

def test_glim_model_init():
    """Test that GLIM model can be initialized with new parameters."""
    print("\n" + "="*80)
    print("Test 3: GLIM model initialization")
    print("="*80)
    
    try:
        from model.glim import GLIM
        
        # Test without sentence embeddings (backward compatibility)
        model = GLIM(
            use_sentence_embeddings=False,
            embed_dim=1024,
        )
        print("✓ GLIM initialized successfully without sentence embeddings")
        
        # Test with sentence embeddings
        model_with_emb = GLIM(
            use_sentence_embeddings=True,
            embed_dim=1024,
        )
        print("✓ GLIM initialized successfully with sentence embeddings")
        
        # Check if projection layer exists
        if hasattr(model_with_emb, 'sentence_emb_proj'):
            print("✓ Sentence embedding projection layer found")
        else:
            print("✗ Sentence embedding projection layer not found")
            return False
            
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_encode_sentence_embeddings():
    """Test the sentence embedding encoding method."""
    print("\n" + "="*80)
    print("Test 4: Sentence embedding encoding")
    print("="*80)
    
    try:
        from model.glim import GLIM
        
        # Initialize model
        model = GLIM(
            use_sentence_embeddings=True,
            embed_dim=1024,
        )
        
        # Create fake sentence embeddings
        batch_size = 4
        sentence_emb = torch.randn(batch_size, 768)
        
        # Encode
        encoded, mask = model.encode_sentence_embeddings(sentence_emb)
        
        # Check shapes
        assert encoded.shape == (batch_size, 1, 1024), f"Expected shape {(batch_size, 1, 1024)}, got {encoded.shape}"
        assert mask.shape == (batch_size, 1), f"Expected mask shape {(batch_size, 1)}, got {mask.shape}"
        
        print(f"✓ Sentence embeddings encoded correctly")
        print(f"  Input shape: {sentence_emb.shape}")
        print(f"  Output shape: {encoded.shape}")
        print(f"  Mask shape: {mask.shape}")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_preprocessing_scripts_imports():
    """Test that preprocessing scripts can be imported."""
    print("\n" + "="*80)
    print("Test 5: Preprocessing scripts imports")
    print("="*80)
    
    success = True
    
    try:
        # Change to data directory
        original_dir = os.getcwd()
        
        # Test imports
        try:
            import data.preprocess_gen_lbl
            print("✓ preprocess_gen_lbl.py imports successfully")
        except Exception as e:
            print(f"✗ preprocess_gen_lbl.py import failed: {e}")
            success = False
        
        try:
            import data.preprocess_mat
            print("✓ preprocess_mat.py imports successfully")
        except Exception as e:
            print(f"✗ preprocess_mat.py import failed: {e}")
            success = False
        
        try:
            import data.preprocess_merge
            print("✓ preprocess_merge.py imports successfully")
        except Exception as e:
            print(f"✗ preprocess_merge.py import failed: {e}")
            success = False
            
    finally:
        os.chdir(original_dir)
    
    return success

def main():
    """Run all validation tests."""
    print("\n" + "="*80)
    print("VALIDATION TESTS FOR SENTENCE EMBEDDING ALIGNMENT CHANGES")
    print("="*80)
    
    results = []
    
    # Run tests
    results.append(("Datamodule without embeddings", test_datamodule_without_embeddings()))
    results.append(("Datamodule with embeddings", test_datamodule_with_embeddings()))
    results.append(("GLIM model initialization", test_glim_model_init()))
    results.append(("Sentence embedding encoding", test_encode_sentence_embeddings()))
    results.append(("Preprocessing scripts imports", test_preprocessing_scripts_imports()))
    
    # Print summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result is True)
    skipped = sum(1 for _, result in results if result is None)
    failed = sum(1 for _, result in results if result is False)
    total = len(results)
    
    for test_name, result in results:
        if result is True:
            status = "✓ PASSED"
        elif result is None:
            status = "⚠ SKIPPED"
        else:
            status = "✗ FAILED"
        print(f"{status:12} {test_name}")
    
    print("-"*80)
    print(f"Total: {total} | Passed: {passed} | Skipped: {skipped} | Failed: {failed}")
    
    if failed == 0:
        print("\n✓ All non-skipped tests passed!")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed!")
        return 1

if __name__ == '__main__':
    exit(main())
