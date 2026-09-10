from types import SimpleNamespace

import pytest

import melody_queue_workflow as workflow


@pytest.mark.parametrize('epochs,skip,train_expected',[(0,False,False),(0,True,False),(10,False,True),(75,True,False)])
def test_zero_skips_only_training_not_review_prediction_or_queue_actions(tmp_path,monkeypatch,epochs,skip,train_expected):
    pair=workflow.StemPair(20,tmp_path/'20_inst.wav',tmp_path/'20_vocal.wav')
    checkpoint=tmp_path/'final.pt'; checkpoint.write_bytes(b'fixture')
    events=[]
    monkeypatch.setattr(workflow,'CHECKPOINT',checkpoint)
    monkeypatch.setattr(workflow,'discover_pairs',lambda *args:[pair])
    monkeypatch.setattr(workflow,'pending_review_rows',lambda *args:[(19,{'status':workflow.STATUS_REVIEW})])
    monkeypatch.setattr(workflow,'prepare_and_accept_reviews',lambda **kwargs:events.append('accept_reviews'))
    monkeypatch.setattr(workflow,'registry_index',lambda *args:{20:{'status':workflow.STATUS_STAGED}})
    monkeypatch.setattr(workflow,'run_command',lambda command:events.append(command[1]))
    monkeypatch.setattr(workflow,'move_predicted_pair_to_dataset',lambda *args,**kwargs:events.append('move_after_prediction'))
    assert workflow.run_next(epochs=epochs,skip_train=skip)==20
    assert events==['accept_reviews']+(['train_melody.py'] if train_expected else [])+['predict_test_audio.py','move_after_prediction']


def test_missing_review_still_blocks_zero_epoch_run(tmp_path,monkeypatch):
    monkeypatch.setattr(workflow,'discover_pairs',lambda *args:[])
    monkeypatch.setattr(workflow,'pending_review_rows',lambda *args:[(19,{})])
    def reject(**kwargs):
        raise FileNotFoundError('review MIDI missing')
    monkeypatch.setattr(workflow,'prepare_and_accept_reviews',reject)
    monkeypatch.setattr(workflow,'run_command',lambda *args:pytest.fail('Must not predict or train'))
    with pytest.raises(FileNotFoundError,match='review MIDI missing'):
        workflow.run_next(epochs=0)


def test_negative_epoch_rejected_before_any_work(monkeypatch):
    monkeypatch.setattr(workflow,'discover_pairs',lambda *args:pytest.fail('No queue work permitted'))
    with pytest.raises(ValueError,match='不能小于 0'):
        workflow.run_next(epochs=-1)


def test_ui_zero_confirmation_says_skip_training(monkeypatch):
    import melody_queue_workflow_ui as ui
    app=ui.MelodyQueueApp.__new__(ui.MelodyQueueApp)
    app.epochs=SimpleNamespace(get=lambda:0)
    app.skip_train=SimpleNamespace(get=lambda:False)
    commands=[]; prompts=[]
    app.start_command=commands.append
    monkeypatch.setattr(ui.messagebox,'askyesno',lambda title,text:prompts.append(text) or True)
    app.run_next()
    assert commands==[['next','--epochs','0']]
    assert '使用现有 checkpoint 直接预测' in prompts[0]
    assert '先训练 0 轮' not in prompts[0]
