"""
下載 US 1999-2018 破產資料（american_bankruptcy_dataset.csv）到 data/raw/bankruptcy/。
來源：GitHub sowide/bankruptcy_dataset
"""
from pathlib import Path
import urllib.request
import sys

project_root = Path(__file__).resolve().parent.parent.parent
out_dir = project_root / "data" / "raw" / "bankruptcy"
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "american_bankruptcy_dataset.csv"
url = "https://raw.githubusercontent.com/sowide/bankruptcy_dataset/main/american_bankruptcy_dataset.csv"

def main():
    print(f"下載中: {url}")
    print(f"存到: {out_file}")
    try:
        urllib.request.urlretrieve(url, out_file)
    except Exception as e:
        print(f"下載失敗: {e}", file=sys.stderr)
        print("請手動從 https://github.com/sowide/bankruptcy_dataset 下載 american_bankruptcy_dataset.csv 放到 data/raw/bankruptcy/")
        sys.exit(1)
    size_mb = out_file.stat().st_size / (1024 * 1024)
    print(f"完成。檔案大小: {size_mb:.2f} MB")
    # 快速檢查前幾行
    with open(out_file, "r", encoding="utf-8", errors="replace") as f:
        head = "".join(f.readline() for _ in range(3))
    print("前幾行預覽:")
    print(head)

if __name__ == "__main__":
    main()
