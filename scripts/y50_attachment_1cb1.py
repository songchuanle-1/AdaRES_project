"""Attach the fixed joint-incidence path to the final indicator call only."""
class FinalJointIncidenceAttachment:
    def __init__(self, decoder, readout):
        self.decoder, self.readout = decoder, readout
        self.enabled = False
        self.wrong_correspondence = False
        self.cached = []
        self.capture = {}
        self.handles = [decoder.register_forward_pre_hook(self.before), decoder.indi_embedding.register_forward_hook(self.indicator)]

    def before(self, module, inputs):
        self.capture = dict(calls=0)
        self.cached = []

    def indicator(self, module, inputs, output):
        self.capture['calls'] += 1
        if not self.enabled or self.capture['calls'] != self.decoder.num_layer + 1:
            return output
        query, masks, padding, masses = inputs
        values = joint_incidence_descriptor(masks, padding, masses)
        if self.wrong_correspondence:
            values = values.roll(1, dims=1)
        residual = self.readout(query, values)
        self.cached = [(q.detach().clone(), d.detach().clone()) for q, d in zip(query, values)]
        return output + torch.stack((torch.zeros_like(residual), residual), dim=-1)

    def close(self):
        for handle in self.handles:
            handle.remove()
