net = resnet50(weights="IMAGENET1K_V2")
net.fc = Identity()

net.to(device)
#ho ridotto la dimensione della lista dei centroidi da 10 a 3
model = PrototypicalNetworks(net, 3, isCuda = True).cuda()

# Load the training dataset
root="/content/sample_data/dataset2/training" #modifica

# Resize the images and transform to tensors
transformation = transforms.Compose([transforms.Resize((160, 160)), #150,150
                                     transforms.RandomHorizontalFlip(), 
                                     transforms.ToTensor()
                                     ])

# Initialize the network
prototypical_dataset = PrototipicalDataset(root, transformation)


# Load the training dataset
#15
my_batch_size = 16
train_dataloader = DataLoader(prototypical_dataset, shuffle=False, num_workers=0, batch_size=my_batch_size)

criterion = nn.CrossEntropyLoss()
criterionContrastive = ContrastiveLoss()
#cambiato learning rate prima era 0.0005
optimizer = optim.Adam(model.parameters(), lr=0.00005)
# optimizer = optim.SGD(model.parameters(), lr=0.0005)

all_loss = []
counter = []
loss_history = []
iteration_number = 0
start_proto = False

model.backbone.train()
model.train()
# Iterate throught the epochs
for epoch in range(5):
  # Iterate over batches
  for i, (img0, img1) in enumerate(train_dataloader, 0):
    sup_dim = 6 #6
    que_dim = 10 #10

    #trasformo in .cuda()
    img0, img1 = img0.cuda(), img1.cuda()

    # Zero the gradients
    optimizer.zero_grad()

    # Pass in the two images into the network and obtain two outputs
    out_scores, query_labels, outReal, outFake = model(img0, img1, sup_dim, que_dim)
   
    # Pass the outputs of the networks and label into the loss function
    b, c = outReal.size()
    labelContrastive = torch.tensor([1] * b)
    loss_contrastive = criterionContrastive(outReal, outFake, labelContrastive.cuda())
    
    real_batch1 = outReal[:int(my_batch_size/2)]
    real_batch2 = outReal[int(my_batch_size/2):]
    labelReal = torch.tensor([0] * int(my_batch_size/2)).cuda()
    loss_real = criterionContrastive(real_batch1, real_batch2, labelReal)

    fake_batch1 = outFake[:int(my_batch_size/2)]
    fake_batch2 = outFake[int(my_batch_size/2):]
    loss_fake = criterionContrastive(fake_batch1, fake_batch2, labelReal)

    lossPrototypical = criterion(out_scores, query_labels.cuda())

    loss = lossPrototypical + loss_contrastive + loss_real + loss_fake

    #backward
    loss.backward()

    # Optimize
    optimizer.step()

    all_loss.append(loss)

    # aggiusto il prototipo dopo l'ottimizzazione
    # model.post_opt_prototype(support_images, support_labels)

    # Every 10 batches print out the loss
    if i % 10 == 0 :
      print(f"Epoch number {epoch}\n Batch number {i}\n Current loss: {loss.item()}")
      print(f" Current lossPrototypical: {lossPrototypical.item()}\n")
      print(f" Current loss_contrastive: {loss_contrastive.item()}\n")
      print(f" Current loss_real: {loss_real.item()}\n")
      print(f" Current loss_fake: {loss_fake.item()}\n")

      iteration_number += 10

      counter.append(iteration_number)
      loss_history.append(loss.item())
  
      


print(f"Epoch number {epoch}\n Batch number {i}\n Current loss: {loss.item()}")
print(f" Current lossPrototypical: {lossPrototypical.item()}\n")
print(f" Current loss_contrastive: {loss_contrastive.item()}\n")
print(f" Current loss_real: {loss_real.item()}\n")
print(f" Current loss_fake: {loss_fake.item()}\n")
show_plot(counter, loss_history)
print("END TRAINING")
