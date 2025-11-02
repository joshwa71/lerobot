# list_libero_tasks.py
from libero.libero import benchmark

def list_suite(suite_name="libero_10"):
    suite = benchmark.get_benchmark_dict()[suite_name]()
    for i, t in enumerate(suite.tasks):
        print(f"{i:02d} | name={t.name} | language={t.language}")

if __name__ == "__main__":
    list_suite("libero_10")