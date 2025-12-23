from src.amlsg_engine import AMLSGEngine

def execute_hybrid_reconstruction(hybrid_data, targets):
    """
    Stage III: Hybrid Anchor Strategy.
    Uses local PM2.5 (Anchor) and Global City Baseline (Context) to reconstruct full history.
    """
    print(">>> Starting Stage III: Hybrid Anchor Reconstruction...")
    
    for target in targets:
        # Configuration for the most complex reconstruction task
        engine = AMLSGEngine(target_label=target, mode='high_fidelity')
        engine.fit(hybrid_data, time_limit=1200) # Longer training for high precision
        print(f"Hybrid model for {target} ready.")
