from tqdm import tqdm
import torch



def train_fn(model, data_loader, optimizer, device, loss_fn, input_key="image", target_key="targets"):
    """
    トレーニング関数
    Args:
        model: モデル
        data_loader: データローダー
        optimizer: 最適化関数
        device: 使用するデバイス（例: 'cuda'）
        loss_fn: 損失関数
        input_key: 入力データのキー（デフォルト: "image"）
        target_key: ラベルデータのキー（デフォルト: "targets"）
    """
    fin_loss = 0
    model.train()
    tk = tqdm(data_loader, total=len(data_loader))
    
    for batch in tk:
        # データをデバイスに転送
        inputs = batch[input_key].to(device)
        targets = batch[target_key].to(device)

        # 勾配の初期化
        optimizer.zero_grad()

        # モデルの順伝播
        outputs = model(inputs)

        # 損失計算
        loss = loss_fn(outputs, targets)

        # 逆伝播と最適化
        loss.backward()
        optimizer.step()

        fin_loss += loss.item()

    return fin_loss / len(data_loader)

def evaluate(model, data_loader, device, loss_fn, input_key="image", target_key="targets"):
    """
    評価関数
    Args:
        model: モデル
        data_loader: データローダー
        device: 使用するデバイス（例: 'cuda'）
        loss_fn: 損失関数
        input_key: 入力データのキー（デフォルト: "image"）
        target_key: ラベルデータのキー（デフォルト: "targets"）
    """
    model.eval()
    fin_loss = 0
    fin_preds = []
    
    with torch.no_grad():
        tk = tqdm(data_loader, total=len(data_loader))
        
        for batch in tk:
            # データをデバイスに転送
            inputs = batch[input_key].to(device)
            targets = batch[target_key].to(device)

            # モデルの順伝播
            outputs = model(inputs)

            # 損失計算
            loss = loss_fn(outputs, targets)

            fin_loss += loss.item()
            fin_preds.append(outputs.cpu())

    return torch.cat(fin_preds, dim=0), fin_loss / len(data_loader)


def predict(model, data_loader, device, input_key="image"):
    """
    予測関数
    Args:
        model: モデル
        data_loader: データローダー
        device: 使用するデバイス（例: 'cuda'）
        input_key: 入力データのキー（デフォルト: "image"）
    """
    model.eval()
    final_predictions = []
    
    with torch.no_grad():
        tk = tqdm(data_loader, total=len(data_loader))
        
        for batch in tk:
            # データをデバイスに転送
            inputs = batch[input_key].to(device)

            # モデルの順伝播
            predictions = model(inputs)
            final_predictions.append(predictions.cpu())

    return torch.cat(final_predictions, dim=0)
