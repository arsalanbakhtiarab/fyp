{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "127dc4f3-8bb8-47c0-93d2-edc1b1916bfb",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "data = pd.read_csv(r\"D:\\FYP\\Final Model Training\\Data Set\\Final_data_set_shuffle_for_model.csv\")\n",
    "# data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "7be8526b-1778-4402-9a4d-ce9de8bcec7d",
   "metadata": {},
   "outputs": [],
   "source": [
    "data = data[['Headline','Label']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "7fdc8f52-28f2-4534-aa31-67dbf0cb9944",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score\n",
    "import torch\n",
    "from transformers import TrainingArguments,Trainer\n",
    "from transformers import BertTokenizer,BertForSequenceClassification"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "2d16f4f6-05b5-4322-a907-61620a2b595c",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert/distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']\n",
      "You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.\n"
     ]
    }
   ],
   "source": [
    "from transformers import AutoTokenizer\n",
    "tokenizer = AutoTokenizer.from_pretrained(\"distilbert/distilbert-base-uncased\")\n",
    "\n",
    "from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer\n",
    "model = AutoModelForSequenceClassification.from_pretrained(\n",
    "    \"distilbert/distilbert-base-uncased\", num_labels=2, \n",
    ")\n",
    "model = model.to('cpu') # if CPU is available\n",
    "# model = model.to('cuda') # if GPU is available"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "f46082f2-a07c-44d3-a8c0-b81d1da92ff7",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = list(data['Headline'])\n",
    "y = list(data['Label'])\n",
    "X_train,X_val,y_train,y_val = train_test_split(X,y,test_size = 0.2,stratify=y)\n",
    "X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length= 310)\n",
    "X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=310)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "047ab224-e46b-4c0d-8ae7-57dadbd0ef0f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create torch dataset\n",
    "class Dataset(torch.utils.data.Dataset):\n",
    "    def __init__(self, encodings, labels=None):\n",
    "        self.encodings = encodings\n",
    "        self.labels = labels\n",
    "\n",
    "    def __getitem__(self, idx):\n",
    "        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}\n",
    "        if self.labels:\n",
    "            item[\"labels\"] = torch.tensor(self.labels[idx])\n",
    "        return item\n",
    "\n",
    "    def __len__(self):\n",
    "        return len(self.encodings[\"input_ids\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "a48b919d-4f77-4e0e-b7fa-e6e718fc0ea2",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_dataset = Dataset(X_train_tokenized, y_train)\n",
    "val_dataset = Dataset(X_val_tokenized, y_val)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "93eb1e0c-d005-4787-86ce-bc44f911678e",
   "metadata": {},
   "outputs": [],
   "source": [
    "def compute_metrics(p):\n",
    "    print(type(p))\n",
    "    pred, labels = p\n",
    "    pred = np.argmax(pred, axis=1)\n",
    "\n",
    "    accuracy = accuracy_score(y_true=labels, y_pred=pred)\n",
    "    recall = recall_score(y_true=labels, y_pred=pred)\n",
    "    precision = precision_score(y_true=labels, y_pred=pred)\n",
    "    f1 = f1_score(y_true=labels, y_pred=pred)\n",
    "\n",
    "    return {\"accuracy\": accuracy, \"precision\": precision, \"recall\": recall, \"f1\": f1}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "c2f4b48c-8576-46d8-90ae-69171f866f5e",
   "metadata": {},
   "outputs": [],
   "source": [
    "from transformers import DataCollatorWithPadding\n",
    "data_collator = DataCollatorWithPadding(tokenizer=tokenizer)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "7ec85cc2-04ce-490e-af47-86099f56e696",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']\n",
      "You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.\n"
     ]
    }
   ],
   "source": [
    "from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments\n",
    "\n",
    "training_args = TrainingArguments(\n",
    "    output_dir='./results',          # output directory\n",
    "    num_train_epochs=3,              # total number of training epochs\n",
    "    per_device_train_batch_size=16,  # batch size per device during training\n",
    "    per_device_eval_batch_size=64,   # batch size for evaluation\n",
    "    warmup_steps=500,                # number of warmup steps for learning rate scheduler\n",
    "    weight_decay=0.01,               # strength of weight decay\n",
    "    logging_dir='./logs',            # directory for storing logs\n",
    "    logging_steps=10,\n",
    ")\n",
    "\n",
    "model = DistilBertForSequenceClassification.from_pretrained(\"distilbert-base-uncased\")\n",
    "\n",
    "trainer = Trainer(\n",
    "    model=model,                         # the instantiated  Transformers model to be trained\n",
    "    args=training_args,                  # training arguments, defined above\n",
    "    train_dataset=train_dataset,         # training dataset\n",
    "    eval_dataset=val_dataset             # evaluation dataset\n",
    ")\n",
    "\n",
    "# trainer.train()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "5ded7fe5-5c44-4ef8-8f82-dfdf0e2a2995",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "\n",
       "    <div>\n",
       "      \n",
       "      <progress value='3' max='18657' style='width:300px; height:20px; vertical-align: middle;'></progress>\n",
       "      [    3/18657 00:07 < 36:40:23, 0.14 it/s, Epoch 0.00/3]\n",
       "    </div>\n",
       "    <table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       " <tr style=\"text-align: left;\">\n",
       "      <th>Step</th>\n",
       "      <th>Training Loss</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "  </tbody>\n",
       "</table><p>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "ename": "KeyboardInterrupt",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[37], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m \u001b[43mtrainer\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mtrain\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[1;32m~\\AppData\\Roaming\\Python\\Python312\\site-packages\\transformers\\trainer.py:1624\u001b[0m, in \u001b[0;36mTrainer.train\u001b[1;34m(self, resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)\u001b[0m\n\u001b[0;32m   1622\u001b[0m         hf_hub_utils\u001b[38;5;241m.\u001b[39menable_progress_bars()\n\u001b[0;32m   1623\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m-> 1624\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43minner_training_loop\u001b[49m\u001b[43m(\u001b[49m\n\u001b[0;32m   1625\u001b[0m \u001b[43m        \u001b[49m\u001b[43margs\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43margs\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1626\u001b[0m \u001b[43m        \u001b[49m\u001b[43mresume_from_checkpoint\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mresume_from_checkpoint\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1627\u001b[0m \u001b[43m        \u001b[49m\u001b[43mtrial\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mtrial\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1628\u001b[0m \u001b[43m        \u001b[49m\u001b[43mignore_keys_for_eval\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mignore_keys_for_eval\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1629\u001b[0m \u001b[43m    \u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[1;32m~\\AppData\\Roaming\\Python\\Python312\\site-packages\\transformers\\trainer.py:1961\u001b[0m, in \u001b[0;36mTrainer._inner_training_loop\u001b[1;34m(self, batch_size, args, resume_from_checkpoint, trial, ignore_keys_for_eval)\u001b[0m\n\u001b[0;32m   1958\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mcontrol \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mcallback_handler\u001b[38;5;241m.\u001b[39mon_step_begin(args, \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mstate, \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mcontrol)\n\u001b[0;32m   1960\u001b[0m \u001b[38;5;28;01mwith\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39maccelerator\u001b[38;5;241m.\u001b[39maccumulate(model):\n\u001b[1;32m-> 1961\u001b[0m     tr_loss_step \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mtraining_step\u001b[49m\u001b[43m(\u001b[49m\u001b[43mmodel\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43minputs\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m   1963\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m (\n\u001b[0;32m   1964\u001b[0m     args\u001b[38;5;241m.\u001b[39mlogging_nan_inf_filter\n\u001b[0;32m   1965\u001b[0m     \u001b[38;5;129;01mand\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m is_torch_tpu_available()\n\u001b[0;32m   1966\u001b[0m     \u001b[38;5;129;01mand\u001b[39;00m (torch\u001b[38;5;241m.\u001b[39misnan(tr_loss_step) \u001b[38;5;129;01mor\u001b[39;00m torch\u001b[38;5;241m.\u001b[39misinf(tr_loss_step))\n\u001b[0;32m   1967\u001b[0m ):\n\u001b[0;32m   1968\u001b[0m     \u001b[38;5;66;03m# if loss is nan or inf simply add the average of previous logged losses\u001b[39;00m\n\u001b[0;32m   1969\u001b[0m     tr_loss \u001b[38;5;241m+\u001b[39m\u001b[38;5;241m=\u001b[39m tr_loss \u001b[38;5;241m/\u001b[39m (\u001b[38;5;241m1\u001b[39m \u001b[38;5;241m+\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mstate\u001b[38;5;241m.\u001b[39mglobal_step \u001b[38;5;241m-\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_globalstep_last_logged)\n",
      "File \u001b[1;32m~\\AppData\\Roaming\\Python\\Python312\\site-packages\\transformers\\trainer.py:2911\u001b[0m, in \u001b[0;36mTrainer.training_step\u001b[1;34m(self, model, inputs)\u001b[0m\n\u001b[0;32m   2909\u001b[0m         scaled_loss\u001b[38;5;241m.\u001b[39mbackward()\n\u001b[0;32m   2910\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m-> 2911\u001b[0m     \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43maccelerator\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mbackward\u001b[49m\u001b[43m(\u001b[49m\u001b[43mloss\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m   2913\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m loss\u001b[38;5;241m.\u001b[39mdetach() \u001b[38;5;241m/\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39margs\u001b[38;5;241m.\u001b[39mgradient_accumulation_steps\n",
      "File \u001b[1;32m~\\AppData\\Roaming\\Python\\Python312\\site-packages\\accelerate\\accelerator.py:1966\u001b[0m, in \u001b[0;36mAccelerator.backward\u001b[1;34m(self, loss, **kwargs)\u001b[0m\n\u001b[0;32m   1964\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mscaler\u001b[38;5;241m.\u001b[39mscale(loss)\u001b[38;5;241m.\u001b[39mbackward(\u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n\u001b[0;32m   1965\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m-> 1966\u001b[0m     \u001b[43mloss\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mbackward\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mkwargs\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[1;32m~\\AppData\\Roaming\\Python\\Python312\\site-packages\\torch\\_tensor.py:522\u001b[0m, in \u001b[0;36mTensor.backward\u001b[1;34m(self, gradient, retain_graph, create_graph, inputs)\u001b[0m\n\u001b[0;32m    512\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m has_torch_function_unary(\u001b[38;5;28mself\u001b[39m):\n\u001b[0;32m    513\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m handle_torch_function(\n\u001b[0;32m    514\u001b[0m         Tensor\u001b[38;5;241m.\u001b[39mbackward,\n\u001b[0;32m    515\u001b[0m         (\u001b[38;5;28mself\u001b[39m,),\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m    520\u001b[0m         inputs\u001b[38;5;241m=\u001b[39minputs,\n\u001b[0;32m    521\u001b[0m     )\n\u001b[1;32m--> 522\u001b[0m \u001b[43mtorch\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mautograd\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mbackward\u001b[49m\u001b[43m(\u001b[49m\n\u001b[0;32m    523\u001b[0m \u001b[43m    \u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mgradient\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mretain_graph\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mcreate_graph\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43minputs\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43minputs\u001b[49m\n\u001b[0;32m    524\u001b[0m \u001b[43m\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[1;32m~\\AppData\\Roaming\\Python\\Python312\\site-packages\\torch\\autograd\\__init__.py:266\u001b[0m, in \u001b[0;36mbackward\u001b[1;34m(tensors, grad_tensors, retain_graph, create_graph, grad_variables, inputs)\u001b[0m\n\u001b[0;32m    261\u001b[0m     retain_graph \u001b[38;5;241m=\u001b[39m create_graph\n\u001b[0;32m    263\u001b[0m \u001b[38;5;66;03m# The reason we repeat the same comment below is that\u001b[39;00m\n\u001b[0;32m    264\u001b[0m \u001b[38;5;66;03m# some Python versions print out the first line of a multi-line function\u001b[39;00m\n\u001b[0;32m    265\u001b[0m \u001b[38;5;66;03m# calls in the traceback and some print out the last line\u001b[39;00m\n\u001b[1;32m--> 266\u001b[0m \u001b[43mVariable\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_execution_engine\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mrun_backward\u001b[49m\u001b[43m(\u001b[49m\u001b[43m  \u001b[49m\u001b[38;5;66;43;03m# Calls into the C++ engine to run the backward pass\u001b[39;49;00m\n\u001b[0;32m    267\u001b[0m \u001b[43m    \u001b[49m\u001b[43mtensors\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    268\u001b[0m \u001b[43m    \u001b[49m\u001b[43mgrad_tensors_\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    269\u001b[0m \u001b[43m    \u001b[49m\u001b[43mretain_graph\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    270\u001b[0m \u001b[43m    \u001b[49m\u001b[43mcreate_graph\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    271\u001b[0m \u001b[43m    \u001b[49m\u001b[43minputs\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    272\u001b[0m \u001b[43m    \u001b[49m\u001b[43mallow_unreachable\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43;01mTrue\u001b[39;49;00m\u001b[43m,\u001b[49m\n\u001b[0;32m    273\u001b[0m \u001b[43m    \u001b[49m\u001b[43maccumulate_grad\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43;01mTrue\u001b[39;49;00m\u001b[43m,\u001b[49m\n\u001b[0;32m    274\u001b[0m \u001b[43m\u001b[49m\u001b[43m)\u001b[49m\n",
      "\u001b[1;31mKeyboardInterrupt\u001b[0m: "
     ]
    }
   ],
   "source": [
    "trainer.train()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ca6907c3-6283-487d-9d65-cf871261a6d0",
   "metadata": {},
   "outputs": [],
   "source": [
    "trainer.evaluate()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6360369c-70a4-477e-854d-5f8106d81f48",
   "metadata": {},
   "outputs": [],
   "source": [
    "trainer.save_model(\"Fine-Tuned-google-bert/bert-large-uncased-310-epoch-distilbert/distilbert-base-uncased\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "84c9e249-4fc5-425d-8056-b52c32d89e51",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
