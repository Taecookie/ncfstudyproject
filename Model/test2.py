import os
import sys

print("=== Runtime Path Info ===")

# 1. 현재 작업 디렉터리 (가장 중요)
print("Current Working Directory (cwd):")
print(os.getcwd())

# 2. 실행 중인 파이썬 파일의 실제 위치
if "__file__" in globals():
    print("\nScript File Directory:")
    print(os.path.dirname(os.path.abspath(__file__)))
else:
    print("\nScript File Directory:")
    print("N/A (Interactive environment)")

# 3. 파이썬 실행 파일 위치 (venv 여부 확인 가능)
print("\nPython Executable Path:")
print(sys.executable)

# 4. sys.path (모듈 로딩 기준 경로들)
print("\nPython sys.path:")
for p in sys.path:
    print(p)
