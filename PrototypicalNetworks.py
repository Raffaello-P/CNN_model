class PrototypicalNetworks(nn.Module):
    def __init__(self, backbone: nn.Module, dim_centroid_list, isCuda = False):
        super(PrototypicalNetworks, self).__init__()
        self.backbone = backbone
        #dimensione scelta all'atto dell'inizializzazione della lista di centroid
        self.dim_centroid_list = dim_centroid_list
        #lista di centroidi da cui calcolare quello finale ad ogni batch/epoca
        self.all_centroid = torch.zeros(self.dim_centroid_list, 2, 2048) #2048 - 512
        #numero di centroidi calcolati 
        self.number_centroid = 0
        #centroidi da utilizzare in fase di test
        self.register_buffer("prototype", torch.zeros(2, 2048)) #512 vggface2
        #per decidere se usare internamente .cuda() su prototype nel calcolo della distanza
        self.isCuda = isCuda

    #funzione utilizzata in fase di test 
    def use_the_model(self, query_image: torch.Tensor):
      features = self.backbone(query_image)
      if self.isCuda:
        q_dist = torch.cdist(features, self.prototype.cuda())
      else:
        q_dist = torch.cdist(features, self.prototype)
      return torch.min(q_dist,1).indices
      #argmin

      
      
    def forward(self, imgReal: torch.Tensor, imgFake: torch.Tensor, s_size: int, q_size: int):
      # Creo unico tensore per fare forward
      real_fake_tensor = torch.cat((imgReal, imgFake), 0)
      # Estrazione features 
      features_tensor = self.backbone.forward(real_fake_tensor)
      # indice fine real, inizio fake
      split_tensor = int(features_tensor.shape[0]/2)
      # ricostruisco per contrastive i tensori real e fake
      featuresReal = features_tensor[:split_tensor]
      featuresFake = features_tensor[split_tensor:]
      # costruzione support e query per prototypical
      z_support, support_labels, z_query, query_labels = batch_prototypal_network(featuresReal, featuresFake, s_size, q_size)
      if self.isCuda:
        z_support, support_labels, z_query, query_labels = z_support.cuda(), support_labels.cuda(), z_query.cuda(), query_labels.cuda()
      # Infer the number of different classes from the labels of the support set
      n_way = len(torch.unique(support_labels))
      #print(f"n_way: {n_way}")
      # Prototype i is the mean of all instances of features corresponding to labels == i
      self.all_centroid[self.number_centroid%self.dim_centroid_list] = torch.cat(
        [
            z_support[torch.nonzero(support_labels == label)].mean(0)
            for label in range(n_way)
        ]
        ).detach()
      self.number_centroid = self.number_centroid + 1
      if self.number_centroid < self.dim_centroid_list:
        self.prototype = torch.div(torch.sum(self.all_centroid, 0), self.number_centroid)
      else:
        self.prototype = torch.div(torch.sum(self.all_centroid, 0), self.dim_centroid_list)

      # Compute the euclidean distance from queries to prototypes
      if self.isCuda:
        dists = torch.cdist(z_query, self.prototype.cuda())
      else:
        dists = torch.cdist(z_query, self.prototype)

      # And here is the super complicated operation to transform those distances into classification scores!
      scores = -dists
      return z_support, z_query, support_labels, scores, query_labels, featuresReal, featuresFake



class Identity(nn.Module):
  def __int__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x
