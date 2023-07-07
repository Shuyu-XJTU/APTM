import utils
from train_tools import mlm, attr_loss_mult

def train_pa100k(model, data_loader, optimizer, tokenizer, epoch, device, scheduler, config, mask_generator=None):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itc', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if config['mlm']:
        metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i, (image, label) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        text = ['the person is a man', 'the person is a woman',
                'the person is no more than 60 years old', 'the person is older than 60 years old',
                'the person is a young or old one', 'the person is of mid age, between 18 and 60 years old',
                'the person is older than 18', 'the person is a baby or a teenager, younger than 18',

                'the picture is not the front of the person', 'the picture shows the front of the person',
                'the picture is not the side of the person', 'the picture shows the side of the person',
                'the picture is not the back of the person', 'the picture shows the back of the person',
                'a person without a hat', 'a person with a hat',

                'a person without a glasses', 'a person with a glasses',
                'a person without a handbag', 'a person with a handbag',
                'a person without a shoulder bag', 'a person with a shoulder bag',
                'a person without a backpack', 'a person with a backpack',

                'the person does not hold an object in front', 'the person hold an object in front',
                'the person does not wear short sleeved upper clothes', 'the person wears short sleeved upper clothes',
                'the person does not wear long sleeved upper clothes', 'the person wears long sleeved upper clothes',
                'there is no stride on the upper clothes of the person',
                'there is stride on the upper clothes of the person',

                'there is no logo on the upper clothes of the person',
                'there is logo on the upper clothes of the person',
                'there is no plaid on the upper clothes of the person',
                'there is plaid on the upper clothes of the person',
                'there is no splice on the upper clothes of the person',
                'there is splice on the upper clothes of the person',
                'there is no stripe on the upper clothes of the person',
                'there is stripe on the upper clothes of the person',

                'there is no pattern on the lower part of the person',
                'there is pattern on the lower part of the person',
                'the person does not wear long coat', 'the person wears long coat',
                'the person does not wear trousers', 'the person wears trousers',
                'the person does not wear shorts', 'the person wears shorts',

                'the person does not wear a skirt or a dress', 'the person wears a skirt or a dress',
                'the person does not wear boots', 'the person wears boots',
                ]

        text_input = tokenizer(text, padding='longest', max_length=config['max_tokens'],
                               return_tensors="pt").to(device)

        # mlm loss
        if config['mlm']:
            text_ids_masked, masked_pos, masked_ids = mlm(text, text_input, tokenizer, device, mask_generator, config, True)
            loss_itc, loss_itm, loss_mlm = model(image, text_input.input_ids, text_input.attention_mask,
                                                 text_ids_masked=text_ids_masked, masked_pos=masked_pos,
                                                 masked_ids=masked_ids, label=label)
            loss = loss_itc + loss_itm + loss_mlm
        else:
            loss_itc, loss_itm = model(image, text_input.input_ids, text_input.attention_mask, label=label)
            loss = loss_itc + loss_itm

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        metric_logger.update(loss_itc=loss_itc.item())
        metric_logger.update(loss_itm=loss_itm.item())
        if config['mlm']:
            metric_logger.update(loss_mlm=loss_mlm.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def train_pa100k_only_img_classifier(model, data_loader, optimizer, tokenizer, epoch, device, scheduler, config,
                             mask_generator=None):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i, (image, label) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        loss = model(image, None, None, label=label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}