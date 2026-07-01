#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seed-ProVer Client for Lean 4 Proof Verification

Interfaces with Seed-ProVer agentic architecture to automatically
verify Lean 4 theorems using LLM-guided proof search.

Two modes:
1. API mode: Use Seed-ProVer API (if available)
2. Local mode: Use resilient_llm_client to generate tactics, then verify with Lean
"""

import subprocess
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import tempfile

# Add parent directory to path for importing resilient_llm_client
sys.path.insert(0, str(Path(__file__).parent.parent))
from resilient_llm_client import ResilientLLMClient


class SeedProverClient:
    """Client for Seed-ProVer agentic theorem proving."""
    
    def __init__(self, use_api: bool = False, lean_path: str = "lean"):
        self.use_api = use_api
        self.lean_path = lean_path
        self.llm_client = ResilientLLMClient() if not use_api else None
        
    def verify_lean_file(self, lean_file: str) -> Tuple[bool, str]:
        """Verify a Lean file using Lean 4 compiler.
        
        Returns:
            (success, output) tuple
        """
        try:
            result = subprocess.run(
                [self.lean_path, "--run", lean_file],
                capture_output=True,
                text=True,
                timeout=300
            )
            success = result.returncode == 0
            output = result.stdout + "\n" + result.stderr
            return success, output
        except subprocess.TimeoutExpired:
            return False, "Timeout: Lean verification took too long"
        except FileNotFoundError:
            return False, f"Lean executable not found: {self.lean_path}"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def generate_proof(self, theorem_statement: str, context: str = "") -> str:
        """Generate a proof for a theorem using LLM.
        
        Uses resilient_llm_client as a drop-in replacement for DeepSeek-Prover-V2.
        """
        if self.use_api:
            return self._generate_proof_api(theorem_statement, context)
        else:
            return self._generate_proof_local(theorem_statement, context)
    
    def _generate_proof_local(self, theorem_statement: str, context: str = "") -> str:
        """Generate proof using resilient_llm_client."""
        prompt = f"""You are a Lean 4 theorem proving expert. Given the following theorem statement and context, generate a complete Lean 4 proof.

Context:
{context}

Theorem:
{theorem_statement}

Generate a complete Lean 4 proof with all necessary tactics. Use standard mathlib tactics and lemmas.
"""
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            resp, metrics = self.llm_client.chat(messages)
            proof = resp["choices"][0]["message"]["content"]
            return proof
        except Exception as e:
            print(f"Error generating proof: {e}")
            return ""
    
    def _generate_proof_api(self, theorem_statement: str, context: str = "") -> str:
        """Generate proof using Seed-ProVer API (if available)."""
        # Placeholder for Seed-ProVer API integration
        # TODO: Implement when Seed-ProVer API is available
        raise NotImplementedError("Seed-ProVer API integration not yet implemented")
    
    def verify_theorem(self, lean_code: str, theorem_name: str) -> Dict:
        """Verify a specific theorem in Lean code.
        
        Args:
            lean_code: Complete Lean 4 code including theorem
            theorem_name: Name of the theorem to verify
            
        Returns:
            Dictionary with verification results
        """
        # Write code to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lean', delete=False) as f:
            f.write(lean_code)
            temp_file = f.name
        
        try:
            # Verify the file
            success, output = self.verify_lean_file(temp_file)
            
            result = {
                'theorem': theorem_name,
                'success': success,
                'output': output,
                'temp_file': temp_file
            }
            
            return result
        finally:
            # Cleanup
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def batch_verify(self, lean_file: str) -> List[Dict]:
        """Verify all theorems in a Lean file.
        
        Returns:
            List of verification results for each theorem
        """
        # Parse Lean file to extract theorems
        theorems = self._extract_theorems(lean_file)
        
        results = []
        for theorem_name in theorems:
            # Read the file and verify
            with open(lean_file) as f:
                lean_code = f.read()
            
            result = self.verify_theorem(lean_code, theorem_name)
            results.append(result)
            
            print(f"Theorem: {theorem_name}")
            print(f"  Success: {result['success']}")
            if not result['success']:
                print(f"  Output: {result['output'][:200]}...")
            print()
        
        return results
    
    def _extract_theorems(self, lean_file: str) -> List[str]:
        """Extract theorem names from a Lean file."""
        theorems = []
        
        with open(lean_file) as f:
            for line in f:
                line = line.strip()
                # Look for theorem/lemma declarations
                if line.startswith('theorem ') or line.startswith('lemma '):
                    # Extract name
                    parts = line.split()
                    if len(parts) >= 2:
                        name = parts[1].rstrip(':')
                        theorems.append(name)
        
        return theorems


def main():
    """Main function to demonstrate Seed-ProVer client usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Seed-ProVer Client for Lean 4')
    parser.add_argument('--lean-file', type=str, help='Lean file to verify')
    parser.add_argument('--theorem', type=str, help='Specific theorem to verify')
    parser.add_argument('--use-api', action='store_true', help='Use Seed-ProVer API')
    parser.add_argument('--lean-path', type=str, default='lean', help='Path to Lean executable')
    args = parser.parse_args()
    
    client = SeedProverClient(use_api=args.use_api, lean_path=args.lean_path)
    
    if args.lean_file:
        if args.theorem:
            # Verify specific theorem
            with open(args.lean_file) as f:
                lean_code = f.read()
            result = client.verify_theorem(lean_code, args.theorem)
            print(f"Theorem: {args.theorem}")
            print(f"Success: {result['success']}")
            print(f"Output:\n{result['output']}")
        else:
            # Verify all theorems
            results = client.batch_verify(args.lean_file)
            print(f"\n{'='*60}")
            print(f"Verification Summary:")
            print(f"{'='*60}")
            successful = sum(1 for r in results if r['success'])
            print(f"  Total theorems: {len(results)}")
            print(f"  Successful: {successful}")
            print(f"  Failed: {len(results) - successful}")
    else:
        print("No Lean file specified. Use --lean-file to specify a file.")
        print("\nExample usage:")
        print("  python seed_prover_client.py --lean-file ../lean_proofs/main_theorems.lean")


if __name__ == '__main__':
    main()
