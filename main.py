# main.py: Main script to demonstrate the AI unit test generator
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.generator import generate_tests

def main():
    """Main function to demonstrate the AI unit test generator."""
    print("AI Unit Test Case Generator")
    print("=" * 50)
    print("This system uses CodeT5 trained on MBPP dataset to generate unit tests.")
    print()
    
    # Example function
    function_code = """
def add_numbers(a, b):
    return a + b
"""
    
    print("Input Function:")
    print(function_code)
    
    print("Generated Test Cases:")
    print("-" * 30)
    
    try:
        tests = generate_tests(function_code.strip())
        print(tests)
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have trained the model first by running: python src/trainer.py")
        return
    
    print()
    print("Test generation completed!")
    print()
    print("Usage:")
    print("  • Training: python src/trainer.py")
    print("  • Interactive: python src/generator.py")
    print("  • Example: python src/example.py")

if __name__ == "__main__":
    main()
