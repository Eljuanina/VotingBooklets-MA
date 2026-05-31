import sys
import os
import shutil



# ─── Step 4: Align with sentence swiss bert ────────────────────────────────────────────────────────────
def run_align():
    print("\n" + "="*60)
    print("STEP 4: Aligning multilingual text")
    print("="*60)
    import align_with_ssb_gold
    align_with_ssb_gold.main()


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    steps = {
        "align":    run_align,
    }
    
    # Usage:
    #   python pipeline.py              -> runs all steps
    #   python pipeline.py clean align  -> runs only clean + align
    #   python pipeline.py align        -> runs only align
    requested = sys.argv[1:] if len(sys.argv) > 1 else list(steps.keys())
    
    for step in requested:
        if step not in steps:
            print(f"Unknown step '{step}'. Valid steps: {', '.join(steps)}")
            sys.exit(1)
    
    for step in requested:
        steps[step]()
    
    print("\n" + "="*60)
    print("Pipeline complete.")
    print("="*60)
