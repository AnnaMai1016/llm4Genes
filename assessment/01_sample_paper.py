import os
import re
import shutil
import random
import json


def sanitize_filename(name):
    return re.sub(r'[<>:"/\\|?*]', '_', name)
    
def main():
    meta_file = '/work/hdd/bbkc/ma7/LiteratureReview/TileDrainageReview/Data/Bib/WOS_core/with_pdf_ris.json'
    MinerU_folder = '/work/hdd/bbkc/ma7/LiteratureReview/TileDrainageReview/Data/mineru'
    sample_folder = 'samples'
    
    with open(meta_file, 'r') as f:
        meta_data = json.load(f)
    select = random.sample(meta_data, 50)

    with open(f"{sample_folder}/sample_meta.json", "w") as f:
        json.dump(select, f)

    os.makedirs(f'{sample_folder}/mineru', exist_ok=True)
    for i, paper in enumerate(select):
        if i%10 == 0:
            print(f'{i}/{len(select)}')
        doi = sanitize_filename(paper['doi'])
        if os.path.exists(f"{MinerU_folder}/{doi}"):
            shutil.copytree(f"{MinerU_folder}/{doi}", f"{sample_folder}/mineru/{doi}")

if __name__ == "__main__":
    main()

