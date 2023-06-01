import editdistance
import json
import os


class ANLS(object):
    def __init__(self, result_dir, exp_name, dataset_name):
        super().__init__()
        self.result_dir = result_dir
        self.exp_name = exp_name
        self.dataset_name = dataset_name
        
    def _ls(self, s1, s2):
        s1 = s1.lower().strip()
        s2 = s2.lower().strip()

        nls = editdistance.eval(s1, s2) / max(len(s1), len(s2))
        
        return 0 if nls >= 0.5 else 1 - nls
    
    def _ls_multiple(self, pred, answers):
        return max(self._ls(pred, answer) for answer in answers)

    def load_and_count(self, split="val"):
        save_path = os.path.join(self.result_dir, f"{self.exp_name}__{split}.json")
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                results = json.load(f)
        else:
            results = []
        return len(results)

    def load_and_save_docvqa(self, qids, questions, predictions, references=None, split="val"):
        all_anls = 0.0
        results = []
        save_path = os.path.join(self.result_dir, f"{self.exp_name}__{split}.json")
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                saved_results = json.load(f)
            for item in saved_results:
                results.append(item)
                all_anls += item.get("anls", 0.0)
        
        for i in range(len(qids)):
            result = {
                "answer": predictions[i],
                "questionId": int(qids[i])
            }
            if references is not None:
                anls = self._ls_multiple(predictions[i], references[i])
                all_anls += anls
                result["question"] = questions[i]
                # result["gold_answer"] = references[i][0]  # select the first answer
                result["gold_answer"] = references[i]  # select the first answer
                result["anls"] = anls                
                
            results.append(result)
        
        with open(save_path, "w") as f:
            json.dump(results, f)
        
        return all_anls / len(results)

    def load_and_save_mpdocvqa(self, qids, questions, predictions, references=None, split="val"):
        all_anls = 0.0
        results = []
        save_path = os.path.join(self.result_dir, f"{self.exp_name}__{split}.json")
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                saved_results = json.load(f)
            for item in saved_results:
                results.append(item)
                all_anls += item.get("anls", 0.0)
        
        for i in range(len(qids)):
            result = {
                "answer": predictions[i],
                "questionId": int(qids[i]),
                "answer_page": 0, # default
            }
            if references is not None:
                anls = self._ls_multiple(predictions[i], references[i])
                all_anls += anls
                result["question"] = questions[i]
                # result["gold_answer"] = references[i][0]  # select the first answer
                result["gold_answer"] = references[i]  # select the first answer
                result["anls"] = anls                
                
            results.append(result)
        
        with open(save_path, "w") as f:
            json.dump(results, f)
        
        return all_anls / len(results)

    def compute_and_save_docvqa(self, qids, questions, predictions, references=None, split="val"):
        all_anls = 0.0
        results = []
        for i in range(len(qids)):
            result = {
                "answer": predictions[i],
                "questionId": int(qids[i])
            }
            if references is not None:
                anls = self._ls_multiple(predictions[i], references[i])
                all_anls += anls
                result["question"] = questions[i]
                # result["gold_answer"] = references[i][0]  # select the first answer
                result["gold_answer"] = references[i]  # select the first answer
                result["anls"] = anls                
                
            results.append(result)
        
        save_path = os.path.join(self.result_dir, f"{self.exp_name}__{split}.json")
        with open(save_path, "w") as f:
            json.dump(results, f)
        
        return all_anls / len(qids)
    
    def compute_and_save_mpdocvqa(self, qids, questions, predictions, references=None, split="val"):
        all_anls = 0.0
        results = []
        for i in range(len(qids)):
            result = {
                "answer": predictions[i],
                "questionId": int(qids[i]),
                "answer_page": 0, # default
            }
            if references is not None:
                anls = self._ls_multiple(predictions[i], references[i])
                all_anls += anls
                result["question"] = questions[i]
                # result["gold_answer"] = references[i][0]  # select the first answer
                result["gold_answer"] = references[i]  # select the first answer
                result["anls"] = anls                
                
            results.append(result)
        
        save_path = os.path.join(self.result_dir, f"{self.exp_name}__{split}.json")
        with open(save_path, "w") as f:
            json.dump(results, f)
        
        return all_anls / len(qids)
    
    def load_and_save(self, qids, questions, predictions, references=None, split="val"):
        os.makedirs(self.result_dir, exist_ok=True)
        if self.dataset_name in ["docvqa", "infographicvqa"]:
            return self.load_and_save_docvqa(qids, questions, predictions, references, split)
        elif self.dataset_name in ["mpdocvqa"]:
            return self.load_and_save_mpdocvqa(qids, questions, predictions, references, split)
        else:
            raise NotImplementedError

    def compute_and_save(self, qids, questions, predictions, references=None, split="val"):
        os.makedirs(self.result_dir, exist_ok=True)
        if self.dataset_name in ["docvqa", "infographicvqa"]:
            return self.compute_and_save_docvqa(qids, questions, predictions, references, split)
        elif self.dataset_name in ["mpdocvqa"]:
            return self.compute_and_save_mpdocvqa(qids, questions, predictions, references, split)
        else:
            raise NotImplementedError

if __name__ == "__main__":
    anls_metric = ANLS(
        result_dir="results",
        exp_name="test",
        dataset_name="docvqa"
    )
    
    r = anls_metric.compute_and_save_docvqa(
        qids=["1", "2", "3"],
        questions=["What is the color of the sky?", "What is the color of the grass?", "What is the color of the water?"],
        predictions=["blue", "green", "blue"],
        references=[["blue"], ["green", "Green"], ["blue", "Blue"]]
    )
    print(r)

    r = anls_metric.load_and_save_docvqa(
        qids=["4", "5", "6"],
        questions=["What is the color of the sky?", "What is the color of the grass?", "What is the color of the water?"],
        predictions=["blue", "green", "blue"],
        references=[["blue"], ["green", "Green"], ["blue", "Blue"]]
    )
    print(r)