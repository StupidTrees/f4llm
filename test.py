from datasets import load_dataset

data = load_dataset("allenai/reward-bench",
                    split="filtered", cache_dir="/userhome/data/llm_cache/")
data.to_json('//userhome/jp/f4llm/data/rewardben.json', orient='records', lines=True)
