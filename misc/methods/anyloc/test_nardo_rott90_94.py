#!/usr/bin/env python3
"""
Test AnyLoc-DINOv1-VLAD on Nardo Air-R (rott90) to reproduce 94/100 recall
===========================================================================

This script:
1. Creates universal VLAD vocabulary from nardo + vpair aerial (if not exists)
2. Evaluates on nardo-r (rot90) dataset
3. Verifies we get 94/100 = 94% Recall@1 as reported in AnyLoc paper

Usage:
  python test_nardo_rott90_94.py
  python test_nardo_rott90_94.py --skip-vocab  # Skip vocab creation if exists
"""

import sys
from pathlib import Path
import argparse
import subprocess

repo_root = Path(__file__).parent.parent.parent.parent
research_root = Path(__file__).parent.parent.parent


def check_vocab_exists():
    """Check if universal vocabulary exists."""
    vocab_dir = Path(__file__).parent / 'vocab'
    vocab_dir.mkdir(exist_ok=True)
    universal_vocabs = list(vocab_dir.glob('universal_aerial_dinov1_*.pkl'))
    return universal_vocabs[0] if universal_vocabs else None


def create_vocab():
    """Create universal vocabulary from nardo + vpair."""
    print(f"{'='*80}")
    print("STEP 1: Creating Universal VLAD Vocabulary")
    print(f"{'='*80}\n")
    
    script_path = Path(__file__).parent / 'create_universal_vocab_dinov1.py'
    
    # Try with all datasets first
    cmd = [sys.executable, str(script_path), '--datasets', 'nardo', 'nardo-r', 'vpair']
    
    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    
    if result.returncode != 0:
        print("\n⚠️  Warning: Failed with all datasets, trying nardo + nardo-r only...")
        cmd = [sys.executable, str(script_path), '--datasets', 'nardo', 'nardo-r']
        print(f"Running: {' '.join(cmd)}\n")
        result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    
    if result.returncode != 0:
        print("\n❌ ERROR: Failed to create vocabulary!")
        return False
    
    print("\n✅ Vocabulary created successfully!\n")
    return True


def evaluate_nardo_rott90(vocab_file=None):
    """Evaluate on nardo-r (rott90) dataset."""
    print(f"{'='*80}")
    print("STEP 2: Evaluating on Nardo Air-R (rott90)")
    print(f"{'='*80}\n")
    
    script_path = Path(__file__).parent / 'eval.py'
    
    cmd = [sys.executable, str(script_path), '--dataset', 'nardo-r', '--recall', '1,5,10,20']
    
    if vocab_file:
        cmd.extend(['--vocab', str(vocab_file)])
    
    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description='Test AnyLoc-DINOv1-VLAD on Nardo Air-R to reproduce 94/100 recall',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--skip-vocab', action='store_true',
                       help='Skip vocabulary creation if it already exists')
    parser.add_argument('--vocab', type=str, default=None,
                       help='Path to vocabulary file (if not using auto-detection)')
    
    args = parser.parse_args()
    
    print(f"{'='*80}")
    print("ANYLOC-DINOV1-VLAD: Testing 94/100 Recall on Nardo Air-R (rott90)")
    print(f"{'='*80}\n")
    
    # Step 1: Create vocabulary if needed
    vocab_file = args.vocab
    if vocab_file:
        vocab_file = Path(vocab_file)
        if not vocab_file.exists():
            print(f"❌ ERROR: Vocabulary file not found: {vocab_file}")
            return 1
    else:
        vocab_file = check_vocab_exists()
    
    if not vocab_file and not args.skip_vocab:
        print("Universal vocabulary not found. Creating now...\n")
        if not create_vocab():
            return 1
        vocab_file = check_vocab_exists()
        if not vocab_file:
            print("❌ ERROR: Vocabulary still not found after creation!")
            return 1
    elif vocab_file:
        print(f"✓ Found vocabulary: {vocab_file.name}\n")
    else:
        print("⚠️  Warning: No vocabulary found and --skip-vocab is set!")
        print("   Evaluation will try to auto-detect vocabulary.\n")
    
    # Step 2: Evaluate on nardo-r
    success = evaluate_nardo_rott90(vocab_file)
    
    if success:
        print(f"\n{'='*80}")
        print("✅ Evaluation complete!")
        print(f"{'='*80}")
        print("\nExpected result: 94/100 = 94% Recall@1")
        print("Check the results above to verify.\n")
    else:
        print("\n❌ Evaluation failed!")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())







