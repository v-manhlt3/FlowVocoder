def build_model(args_dict):
    from models import flowvocoder as waveflow

    from types import SimpleNamespace
    for key, val in args_dict.items():
        if val == 'true' or val == 'True':
            args_dict[key] = True
        elif val == 'false' or val == 'False':
            args_dict[key] = False
    args = SimpleNamespace(**args_dict)


    print('loading FlowVocoder model...')
    model = waveflow.WaveFlow(in_channel=1,
                                  cin_channel=args.cin_channels,
                                  res_channel=args.res_channels,
                                  n_height=args.n_height,
                                  n_flow=args.n_flow,
                                  n_layer=args.n_layer,
                                  layers_per_dilation_h_cycle=args.n_layer_per_cycle,
                                  bipartize=True,
                                  size_flow_embed=args.size_flow_embed
                                  )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("model built: number of parameters: {}".format(total_params))
    model = model.cuda()
    return model