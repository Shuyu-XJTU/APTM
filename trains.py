import utils
from train_tools import mlm
import numpy as np
from thop import profile


def train(model, data_loader, optimizer, tokenizer, epoch, device, scheduler, config, mask_generator=None):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itc', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if config['mlm']:
        metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    if config['eda']:
        for i, (image, text, text_eda, idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            image = image.to(device, non_blocking=True)
            idx = idx.to(device, non_blocking=True)
            # text_input = tokenizer(text, padding='longest', max_length=config['max_tokens'],
            #                        return_tensors="pt").to(device)
            text_input = tokenizer(text, padding='max_length', truncation=True, max_length=config['max_tokens'],
                                   return_tensors="pt").to(device)
            text_input_eda = tokenizer(text_eda, padding='max_length', truncation=True, max_length=config['max_tokens'],
                                       return_tensors="pt").to(device)
            if config['mlm']:
                text_ids_masked, masked_pos, masked_ids = mlm(text, text_input, tokenizer, device, mask_generator,
                                                              config)
                loss_itc, loss_itm, loss_mlm = model(image, text_input.input_ids, text_input.attention_mask,
                                                     text_ids_masked=text_ids_masked,
                                                     masked_pos=masked_pos, masked_ids=masked_ids, idx=idx,
                                                     text_ids_eda=text_input_eda.input_ids,
                                                     text_atts_eda=text_input_eda.attention_mask)
                loss = loss_itc + loss_itm + loss_mlm
            else:
                loss_itc, loss_itm = model(image, text_input.input_ids, text_input.attention_mask, idx=idx,
                                           text_ids_eda=text_input_eda.input_ids,
                                           text_atts_eda=text_input_eda.attention_mask)
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
    else:
        for i, (image, text, idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            image = image.to(device, non_blocking=True)
            idx = idx.to(device, non_blocking=True)
            # text_input = tokenizer(text, padding='longest', max_length=config['max_tokens'],
            #                        return_tensors="pt").to(device)
            text_input = tokenizer(text, padding='max_length', truncation=True, max_length=config['max_tokens'],
                                   return_tensors="pt").to(device)
            # mlm loss
            if config['mlm']:
                text_ids_masked, masked_pos, masked_ids = mlm(text, text_input, tokenizer, device, mask_generator,
                                                              config)
                loss_itc, loss_itm, loss_mlm = model(image, text_input.input_ids,
                                                     text_input.attention_mask,
                                                     text_ids_masked=text_ids_masked,
                                                     masked_pos=masked_pos, masked_ids=masked_ids,
                                                     idx=idx)
                loss = loss_itc + loss_itm + loss_mlm
            else:
                loss_itc, loss_itm = model(image, text_input.input_ids, text_input.attention_mask, idx=idx)
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


def train_attr(model, data_loader, optimizer, tokenizer, epoch, device, scheduler, config, mask_generator=None):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itc', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if config['mlm']:
        metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_attr', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i, (image, text, idx, label) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)
        # text_input = tokenizer(text, padding='longest', max_length=config['max_tokens'],
        #                        return_tensors="pt").to(device)
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=config['max_tokens'],
                               return_tensors="pt").to(device)
        label = label.to(device, non_blocking=True)

        attr = ['the person is a woman', 'the person is a man',
                'the person is younger than 18 years old', 'the person is older than 18 years old',

                'the person with short hair', 'the person with long hair',
                'the person with a hat', 'the person without a hat',
                'the person with a backpack', 'the person without a backpack',
                'the person with a handbag', 'the person without a handbag',
                'the person with a bag', 'the person without a bag',

                'the person wears long sleeved upper clothes', 'the person wears short sleeved upper clothes',
                'the person wears long dress or long pants', 'the person wears short dress or short pants',
                'the person wears dress or skirt', 'the person wears pants or shorts',

                'the person wears black upper clothes', 'the person does not wear black upper clothes',
                'the person wears white upper clothes', 'the person does not wear white upper clothes',
                'the person wears red upper clothes', 'the person does not wear red upper clothes',
                'the person wears purple upper clothes', 'the person does not wear purple upper clothes',

                'the person wears yellow upper clothes', 'the person does not wear yellow upper clothes',
                'the person wears blue upper clothes', 'the person does not wear blue upper clothes',
                'the person wears green upper clothes', 'the person does not wear green upper clothes',
                'the person wears gray upper clothes', 'the person does not wear gray upper clothes',

                'the person wears black lower clothes', 'the person does not wear black lower clothes',
                'the person wears white lower clothes', 'the person does not wear white lower clothes',
                'the person wears purple lower clothes', 'the person does not wear purple lower clothes',
                'the person wears yellow lower clothes', 'the person does not wear yellow lower clothes',

                'the person wears blue lower clothes', 'the person does not wear blue lower clothes',
                'the person wears green lower clothes', 'the person does not wear green lower clothes',
                'the person wears pink lower clothes', 'the person does not wear pink lower clothes',
                'the person wears gray lower clothes', 'the person does not wear gray lower clothes',
                'the person wears brown lower clothes', 'the person does not wear brown lower clothes',

                ]
        attr_input = tokenizer(attr, padding='longest', max_length=config['max_tokens'],
                               return_tensors="pt").to(device)

        # mlm loss
        if config['mlm']:
            text_ids_masked, masked_pos, masked_ids = mlm(text, text_input, tokenizer, device, mask_generator,
                                                          config)
            attr_text_ids_masked, attr_masked_pos, attr_masked_ids = mlm(attr, attr_input, tokenizer, device,
                                                                         mask_generator, config,
                                                                         True)

            loss_itc, loss_itm, loss_mlm, loss_attr = model(image, text_input.input_ids, text_input.attention_mask,
                                                            text_ids_masked=text_ids_masked, masked_pos=masked_pos,
                                                            masked_ids=masked_ids, idx=idx,
                                                            attr_text_ids=attr_input.input_ids,
                                                            attr_text_atts=attr_input.attention_mask,
                                                            attr_text_ids_masked=attr_text_ids_masked,
                                                            attr_masked_pos=attr_masked_pos,
                                                            attr_masked_ids=attr_masked_ids, label=label)
            loss = loss_itc + loss_itm + loss_mlm + config['t'] * loss_attr
        else:
            loss_itc, loss_itm, loss_attr = model(image, text_input.input_ids, text_input.attention_mask, idx=idx,
                                                  attr_text_ids=attr_input.input_ids,
                                                  attr_text_atts=attr_input.attention_mask,
                                                  label=label)
            loss = loss_itc + loss_itm + config['t'] * loss_attr

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        metric_logger.update(loss_itc=loss_itc.item())
        metric_logger.update(loss_itm=loss_itm.item())
        if config['mlm']:
            metric_logger.update(loss_mlm=loss_mlm.item())
        metric_logger.update(loss_attr=loss_attr.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}