# -*- coding : utf-8-*-


class SavedModelDetails:

    def __init__(self, label_encoder, percentile_threshold):
        self.label_encoder = label_encoder
        self.percentile_threshold = percentile_threshold

    def get_label_encoder(self):
        return self.label_encoder

    def get_percentile_threshold(self):
        return self.percentile_threshold
