#If nn training used
import torch

def train(data_loader, model, optimizer, loss_fn, device):
	running_loss = 0.0
	correct = 0
	total = 0
	model.train()
	for x_loader, y_loader in data_loader:
		x_loader, y_loader = x_loader.to(device), y_loader.to(device)

		optimizer.zero_grad()
		
		output = model(x_loader).squeeze(dim=-1)

		loss = loss_fn(output, y_loader)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()

		total += y_loader.size(0)
		correct += (torch.round(output) == y_loader).sum().item()

	accuracy = 100 * correct / total	

	return running_loss, accuracy

def test(data_loader, model, loss_fn, device):
	running_loss = 0.0
	correct = 0
	total = 0
	model.eval()

	y_true = torch.tensor([], dtype=torch.long, device=device)
	y_pred = torch.tensor([], device=device)

	with torch.no_grad():
		for x_loader, y_loader in data_loader:
			x_loader, y_loader = x_loader.to(device), y_loader.to(device)

			output = model(x_loader).squeeze(dim=-1)

			loss = loss_fn(output, y_loader)

			running_loss += loss.item()

			total += y_loader.size(0)
			correct += (torch.round(output) == y_loader).sum().item()

			y_true = torch.cat((y_true, y_loader), 0)
			y_pred = torch.cat((y_pred, torch.round(output)), 0)

	y_true = y_true.cpu().numpy()
	y_pred = y_pred.cpu().numpy()

	accuracy = 100 * correct / total

	return running_loss, accuracy, y_true, y_pred