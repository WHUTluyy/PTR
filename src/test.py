args = get_args_parser()
tokenizer = get_tokenizer(special=[])
temps = get_temps(tokenizer)
train_dataset = REPromptDataset.load(
    path = args.output_dir,
    name = "train",
    temps = temps,
    tokenizer = tokenizer,
    rel2id = args.data_dir + "/" + "rel2id.json")
