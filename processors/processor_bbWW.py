#!/usr/bin/env python

import sys
import os
import datetime
import numpy as np
import logging
import copy
import yaml

import uproot
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.jetmet_tools import CorrectedMETFactory
import cachetools
from coffea import processor
import hist
from coffea import lookup_tools
from coffea.analysis_tools import Weights
from coffea.lumi_tools import LumiMask
from coffea.processor import PackedSelection   ### to move to analysis_tools
import correctionlib


### from TAMU
from coffea.nanoevents.methods import vector
from functools import reduce

class analysis(processor.ProcessorABC):
    def __init__(self, corrections_metadata='analysis/metadata/corrections.yml'):

        self.corrections_metadata = yaml.safe_load(open(corrections_metadata, 'r'))

        # defining histograms
        dataset_axis = hist.axis.StrCategory([], name="dataset", growth=True)
        sel_axis = hist.axis.StrCategory([], name="selection", growth=True)
        pt_axis = hist.axis.Regular(17, 8, 25, name="pt", label=r'$p_{T}$ [GeV]')
        eta_axis = hist.axis.Regular(20, -5, 5, name="eta", label=r'$\eta$')
        phi_axis = hist.axis.Regular(80, -4, 4, name="phi", label=r'$\phi$')
        dR_axis = hist.axis.Regular(100, -50, 50, name="dR", label=r'$\dR$')
        mass_axis = hist.axis.Regular(60, 0, 240, name="mass", label=r'Mass [GeV]')
        pileup_axis = hist.axis.Regular(100, 0, 1, name="puId", label=r'pileup ID')
        jetId_axis = hist.axis.Regular(20, 0, 20, name="jetId", label=r'jet ID')
        beta_axis = hist.axis.Regular(100, 0, 1, name="puId_beta", label=r'beta')
        dR2_axis = hist.axis.Regular(100, 0, 0.1, name="puId_dR2", label=r'dR2Mean')
        frac01_axis = hist.axis.Regular(100, 0, 1, name="puId_frac01", label=r'frac01')
        frac02_axis = hist.axis.Regular(100, 0, 1, name="puId_frac02", label=r'frac02')
        frac03_axis = hist.axis.Regular(100, 0, 1, name="puId_frac03", label=r'frac03')
        frac04_axis = hist.axis.Regular(100, 0, 1, name="puId_frac04", label=r'frac04')
        jetR_axis = hist.axis.Regular(100, 0, 1, name="puId_jetR", label=r'jetR')
        jetRchg_axis = hist.axis.Regular(100, 0, 1, name="puId_jetRchg", label=r'jetRchg')
        majW_axis = hist.axis.Regular(50, 0, 0.5, name="puId_majW", label=r'majW')
        minW_axis = hist.axis.Regular(50, 0, 0.5, name="puId_minW", label=r'minW')
        nCharged_axis = hist.axis.Regular(50, 0, 50, name="puId_nCharged", label=r'nCharged')
        ptD_axis = hist.axis.Regular(100, 0, 2 , name="puId_ptD", label=r'ptD')
        pull_axis = hist.axis.Regular(100, 0, 0.1, name="puId_pull", label=r'pull')
        chEmEF_axis = hist.axis.Regular(100, 0, 1, name="chEmEF", label=r'chEmEF')
        chHEF_axis = hist.axis.Regular(100, 0, 1, name="chHEF", label=r'chHEF')
        muEF_axis = hist.axis.Regular(100, 0, 1, name="muEF", label=r'muEF')
        nConstituents_axis = hist.axis.Regular(80, 0, 80, name="nConstituents", label=r'nConstituents')
        neEmEF_axis = hist.axis.Regular(100, 0, 1, name="neEmEF", label=r'neEmEF')
        neHEF_axis = hist.axis.Regular(100, 0, 1, name="neHEF", label=r'neHEF')


        self.cutflow = {}
        self.hists = {
            'numJets': hist.Hist( dataset_axis, sel_axis,
                                 hist.axis.Regular(15, 0,15, name="numJets", label='Number of jets'),
                                 hist.axis.Regular(15, 0,15, name="numBjets", label='Number of b jets'),
                                 hist.axis.Regular(15, 0,15, name="numnonBjets", label='Number of non b jets'),
                                 ),
            'jet1': hist.Hist( dataset_axis, sel_axis, pt_axis, eta_axis,mass_axis, storage="weight", label="Counts" ),
            'jet1x': hist.Hist(phi_axis, neHEF_axis, label="Counts" ),
            'jet1pileup1': hist.Hist(beta_axis, dR2_axis,frac01_axis, label="Counts" ),
            'jet1pileup2': hist.Hist(frac02_axis,frac03_axis, frac04_axis, label="Counts" ),
            'jet1pileup3': hist.Hist(jetR_axis, jetRchg_axis, majW_axis, label="Counts" ),
            'jet1pileup4': hist.Hist(minW_axis, nCharged_axis, ptD_axis, label="Counts" ),
            'jet1pileup5': hist.Hist(pull_axis, chEmEF_axis, chHEF_axis, label="Counts" ),
            'jet1pileup6': hist.Hist(muEF_axis, nConstituents_axis, neEmEF_axis, label="Counts" ),
            'jet1Ids': hist.Hist(jetId_axis, pileup_axis, label="Counts" ),
            'jet1_': hist.Hist( dataset_axis, sel_axis, pt_axis, eta_axis, mass_axis, storage="weight", label="Counts" ),
            'jet1_x': hist.Hist(phi_axis,neHEF_axis, label="Counts" ),
            'jet1_pileup1': hist.Hist(beta_axis, dR2_axis,frac01_axis, label="Counts" ),
            'jet1_pileup2': hist.Hist(frac02_axis,frac03_axis, frac04_axis, label="Counts" ),
            'jet1_pileup3': hist.Hist(jetR_axis, jetRchg_axis, majW_axis, label="Counts" ),
            'jet1_pileup4': hist.Hist(minW_axis, nCharged_axis, ptD_axis, label="Counts" ),
            'jet1_pileup5': hist.Hist(pull_axis, chEmEF_axis, chHEF_axis, label="Counts" ),
            'jet1_pileup6': hist.Hist(muEF_axis, nConstituents_axis, neEmEF_axis, label="Counts" ),
            'jet1_Ids': hist.Hist(jetId_axis, pileup_axis, label="Counts" ),
            'jet1W': hist.Hist( pt_axis, eta_axis,mass_axis, storage="weight", label="Counts" ),
            'jet1Wx': hist.Hist(phi_axis,neHEF_axis, label="Counts" ),
            'jet1Wpileup1': hist.Hist(beta_axis, dR2_axis,frac01_axis, label="Counts" ),
            'jet1Wpileup2': hist.Hist(frac02_axis,frac03_axis, frac04_axis, label="Counts" ),
            'jet1Wpileup3': hist.Hist(jetR_axis, jetRchg_axis, majW_axis, label="Counts" ),
            'jet1Wpileup4': hist.Hist(minW_axis, nCharged_axis, ptD_axis, label="Counts" ),
            'jet1Wpileup5': hist.Hist(pull_axis, chEmEF_axis, chHEF_axis, label="Counts" ),
            'jet1Wpileup6': hist.Hist(muEF_axis, nConstituents_axis, neEmEF_axis, label="Counts" ),
            'jet1WIds': hist.Hist(jetId_axis, pileup_axis, label="Counts" ),
            'jet2W': hist.Hist( pt_axis, eta_axis,mass_axis, storage="weight", label="Counts" ),
            'jet2Wx': hist.Hist(phi_axis,label="Counts" ),
            'jet2WIds': hist.Hist(dataset_axis,jetId_axis, pileup_axis, label="Counts" ),
            'jet3W': hist.Hist( pt_axis, eta_axis,mass_axis, storage="weight", label="Counts" ),
            'jet3Wx': hist.Hist(phi_axis, neHEF_axis,label="Counts" ),
            'jet3Wpileup1': hist.Hist(beta_axis, dR2_axis,frac01_axis, label="Counts" ),
            'jet3Wpileup2': hist.Hist(frac02_axis,frac03_axis, frac04_axis, label="Counts" ),
            'jet3Wpileup3': hist.Hist(jetR_axis, jetRchg_axis, majW_axis, label="Counts" ),
            'jet3Wpileup4': hist.Hist(minW_axis, nCharged_axis, ptD_axis, label="Counts" ),
            'jet3Wpileup5': hist.Hist(pull_axis, chEmEF_axis, chHEF_axis, label="Counts" ),
            'jet3Wpileup6': hist.Hist(muEF_axis, nConstituents_axis, neEmEF_axis, label="Counts" ),
            'jet3WIds': hist.Hist(dataset_axis, jetId_axis, pileup_axis, label="Counts" ),
            'jet4W': hist.Hist( pt_axis, eta_axis,mass_axis, storage="weight", label="Counts" ),
            'jet4WIds': hist.Hist(dataset_axis, jetId_axis, pileup_axis, storage="weight", label="Counts" ),
            'jet5W': hist.Hist( dataset_axis, sel_axis, pt_axis, eta_axis,mass_axis, storage="weight", label="Counts" ),
            'jet5WIds': hist.Hist(dataset_axis, jetId_axis, pileup_axis, storage="weight", label="Counts" ),
            'jet6W': hist.Hist( dataset_axis, sel_axis, pt_axis, eta_axis,mass_axis, storage="weight", label="Counts" ),
            'jet6WIds': hist.Hist(dataset_axis, jetId_axis, pileup_axis, storage="weight", label="Counts" ),
            
       }


    @property
    def accumulator(self):
        return self.hists

    def process(self, events):

        year = events.metadata['year']

        listOfCuts = PackedSelection()
        isData = False if (events.run[0] == 1) else True
        isMC = ~isData
        logging.info( f'\nProcessing {len(events.event)} events' )

        # adding trigger selection
        trigger_selection = mask_event_decision( events, decision='OR', branch='HLT', list_to_mask=events.metadata['trigger']  )

        # adding golden json filter (just for data)
        lumimask = LumiMask(self.corrections_metadata[year]['goldenJSON'])
        if isData: jsonFilter = np.array( lumimask(events.run, events.luminosityBlock) )
        else: jsonFilter = np.ones( len(events.event), dtype=bool )
        listOfCuts.add( "goldenJson", jsonFilter )

        # adding met filters
        METcleaning = mask_event_decision( events, decision='AND', branch='Flag', list_to_mask=self.corrections_metadata[year]['METFilter'], list_to_skip=['BadPFMuonDzFilter', 'hfNoisyHitsFilter']  )
        listOfCuts.add( "PassesMETFilters", METcleaning )

        ##############################################################
        # From here is a copy of the TAMU code
        # basic object selection
        # from https://github.com/aebid/HHbbWW_Run3/blob/main/python/object_selection.py#L8
        muons = copy.deepcopy(events.Muon)
        muon_conept = ak.where(
            (abs(muons.pdgId) == 13) & (muons.mediumId) & (muons.mvaTTH > 0.50),
                muons.pt,
                0.9 * muons.pt * (1.0 + muons.jetRelIso)
        )

        electrons = copy.deepcopy(events.Electron)
        electron_conept = ak.where(
            (abs(electrons.pdgId) == 11) & (electrons.mvaTTH > 0.30),
                electrons.pt,
                0.9 * electrons.pt * (1.0 + electrons.jetRelIso)
        )

        muons = ak.with_field(muons, muon_conept, "conept")   ### this is stupid
        electrons = ak.with_field(electrons, electron_conept, "conept")

        # from https://github.com/aebid/HHbbWW_Run3/blob/main/python/object_selection.py#L34
        ak4_jets = ak.pad_none(events.Jet, 1)
        ak8_jets = events.FatJet
        ak8_subjets = ak.pad_none(events.SubJet, 2)
        muons = ak.with_field(muons, ak4_jets[muons.jetIdx].mask[(muons.jetIdx >= 0)], "ak4_jets")
        electrons = ak.with_field(electrons, ak4_jets[electrons.jetIdx].mask[(electrons.jetIdx >= 0)], "ak4_jets")
        ak8_jets = ak.with_field(ak8_jets, ak8_subjets[ak8_jets.subJetIdx1].mask[(ak8_jets.subJetIdx1 >= 0)], "subjet1")
        ak8_jets = ak.with_field(ak8_jets, ak8_subjets[ak8_jets.subJetIdx2].mask[(ak8_jets.subJetIdx2 >= 0)], "subjet2")

        jetDeepJet_min_pt = 20.0; jetDeepJet_max_pt = 45.0

        muons.jetDeepJet_Upper_x1 = ak.where(
            (muons.conept - jetDeepJet_min_pt) >= 0.0,
                muons.conept - jetDeepJet_min_pt,
                0.0
        )

        muons.jetDeepJet_Upper_x2 = ak.where(
            (muons.jetDeepJet_Upper_x1/(jetDeepJet_max_pt - jetDeepJet_min_pt)) <= 1.0,
                (muons.jetDeepJet_Upper_x1/(jetDeepJet_max_pt - jetDeepJet_min_pt)),
                1.0
        )

        jetDeepJet_WP_loose = 0.0613 ### this is for 2016
        jetDeepJet_WP_medium = 0.3093 ### this is for 2016
        jetDeepJet_WP_tight = 0.7221 ### this is for 2016
        muons = ak.with_field(muons, muons.jetDeepJet_Upper_x2 * jetDeepJet_WP_loose + (1 - muons.jetDeepJet_Upper_x2) * jetDeepJet_WP_medium, "jetDeepJet_Upper")

        # https://github.com/aebid/HHbbWW_Run3/blob/main/python/object_selection.py#L71
        muon_preselection_mask = (
            (abs(muons.eta) <= 2.4) & (muons.pt >= 5.0) & (abs(muons.dxy) <= 0.05) &
            (abs(muons.dz) <= 0.1) & (muons.miniPFRelIso_all <= 0.4) &
            (muons.sip3d <= 8) & (muons.looseId)
        )

        muon_fakeable_mask = (
            (muons.conept >= 10.0) &
            ak.where(
                ak.is_none(muons.ak4_jets, axis=1),
                    (
                        (muons.mvaTTH > 0.50)
                        | #OR
                        (muons.jetRelIso < 0.80)
                    ),
                    (muons.ak4_jets.btagDeepFlavB <= jetDeepJet_WP_medium) &
                    (
                        (muons.mvaTTH > 0.50)
                        | #OR
                        (muons.jetRelIso < 0.80) & (muons.ak4_jets.btagDeepFlavB <= muons.jetDeepJet_Upper)
                    )
                )
        )

        muon_tight_mask = ((muons.mvaTTH >= 0.50) & (muons.mediumId))

        muons = ak.with_field(muons, muon_preselection_mask, "preselected")
        muons = ak.with_field(muons, muon_fakeable_mask & muon_preselection_mask, "fakeable")
        muons = ak.with_field(muons, muon_tight_mask & muon_fakeable_mask & muon_preselection_mask, "tight")


        muons = ak.with_field(muons, (getattr(muons, 'genPartFlav', False) == 1) | (getattr(muons, 'genPartFlav', False) == 15), "MC_Match")

        # https://github.com/aebid/HHbbWW_Run3/blob/main/python/object_selection.py#L116
        mvaNoIso_WPL = getattr(electrons, 'mvaIso_WP90', False) | getattr(electrons, 'mvaNoIso_WPL', False) | getattr(electrons, 'mvaFall17V2noIso_WPL', False)
        mvaNoIso_WP90 = getattr(electrons, 'mvaIso_WP90', False) | getattr(electrons, 'mvaNoIso_WPL90', False) | getattr(electrons, 'mvaFall17V2noIso_WP90', False)

        electron_preselection_mask = (
            (abs(electrons.eta) <= 2.5) & (electrons.pt >= 7.0)& (abs(electrons.dxy) <= 0.05) &
            (abs(electrons.dz) <= 0.1) & (electrons.miniPFRelIso_all <= 0.4) &
            (electrons.sip3d <= 8) & (mvaNoIso_WPL) & (electrons.lostHits <= 1)
        )

        electron_fakeable_mask = (
            (electrons.conept >= 10.0) & (electrons.hoe <= 0.10) & (electrons.eInvMinusPInv >= -0.04) &
            (electrons.lostHits == 0) & (electrons.convVeto) &
            (
                (abs(electrons.eta + electrons.deltaEtaSC) > 1.479) & (electrons.sieie <= 0.030)
                | #OR
                (abs(electrons.eta + electrons.deltaEtaSC) <= 1.479) & (electrons.sieie <= 0.011)
            ) &
            ak.where(
                ak.is_none(electrons.ak4_jets, axis=1),
                    True,
                    ak.where(
                        electrons.mvaTTH < 0.3,
                            electrons.ak4_jets.btagDeepFlavB <= jetDeepJet_WP_tight,
                            electrons.ak4_jets.btagDeepFlavB <= jetDeepJet_WP_medium
                    )
            ) &
            ak.where(
                electrons.mvaTTH < 0.30,
                    (electrons.jetRelIso < 0.7) & (mvaNoIso_WP90),
                    True
            )
        )

        electron_tight_mask = electrons.mvaTTH >= 0.30

        electrons = ak.with_field(electrons, electron_preselection_mask, "preselected")

        electrons = electrons
        mu_padded = ak.pad_none(muons, 1)
        #We pad muons with 1 None to allow all electrons to have a "pair" for cleaning

        ele_mu_pair_for_cleaning = ak.cartesian(
            [electrons.mask[electrons.preselected], mu_padded.mask[mu_padded.preselected]], nested=True
        )

        ele_for_cleaning, mu_for_cleaning = ak.unzip( ele_mu_pair_for_cleaning )

        electron_cleaning_dr = ak.where(
            (ak.is_none(mu_for_cleaning, axis=2) == 0) & (ak.is_none(ele_for_cleaning, axis=2) == 0),
                abs(ele_for_cleaning.delta_r(mu_for_cleaning)),
                electrons.preselected
        )

        electron_cleaning_mask = ak.min(electron_cleaning_dr, axis=2) > 0.30

        electrons = ak.with_field(electrons, electron_cleaning_mask & electrons.preselected, "cleaned")
        electrons = ak.with_field(electrons, electron_fakeable_mask & electron_cleaning_mask & electron_preselection_mask, "fakeable")
        electrons = ak.with_field(electrons, electron_tight_mask & electron_fakeable_mask & electron_cleaning_mask & electron_preselection_mask, "tight")

        electrons = ak.with_field(electrons, (getattr(electrons, 'genPartFlav', 0) == 1) | (getattr(electrons, 'genPartFlav', 0) == 15), "MC_Match")

        # https://github.com/aebid/HHbbWW_Run3/blob/main/python/object_selection.py#L193
        ##### jet selection
        ak4_jet_preselection_mask = (
            (ak4_jets.pt >= 8.0) & (abs(ak4_jets.eta) <= 2.4) & (ak4_jets.pt <30.0)
        )

        ak4_jets_loose_btag_mask = ak4_jets.btagDeepFlavB > jetDeepJet_WP_loose

        ak4_jets_medium_btag_mask = ak4_jets.btagDeepFlavB > jetDeepJet_WP_medium

        ak4_jets = ak.with_field(ak4_jets, ak4_jet_preselection_mask, "preselected")
        ak4_jets = ak4_jets

        leptons_fakeable = ak.concatenate([electrons.mask[electrons.fakeable], muons.mask[muons.fakeable]], axis=1)
        #argsort fails if all values are none, fix by a all none check
        #Check if variables are none, then if all in the nested list are none, then if all in the unnested list are none
        if not ak.all(ak.all(ak.is_none(leptons_fakeable, axis=1), axis=1)):
            leptons_fakeable = leptons_fakeable[ak.argsort(leptons_fakeable.conept, ascending=False)]
        leptons_fakeable = ak.pad_none(leptons_fakeable, 1)

        ak4_jet_lep_pair_for_cleaning = ak.cartesian([ak4_jets.mask[ak4_jets.preselected], leptons_fakeable], nested=True)
        ak4_jet_for_cleaning, lep_for_cleaning = ak.unzip( ak4_jet_lep_pair_for_cleaning )

        ak4_jet_cleaning_dr_all = ak.fill_none(abs(ak4_jet_for_cleaning.delta_r(lep_for_cleaning)), True)
        ak4_jet_cleaning_mask_all = ak.min(ak4_jet_cleaning_dr_all, axis=2) > 0.40

        ak4_jet_cleaning_dr_single = ak.fill_none(abs(ak4_jet_for_cleaning.delta_r(lep_for_cleaning)), True)[:,:,0:1]
        ak4_jet_cleaning_mask_single = ak.min(ak4_jet_cleaning_dr_single, axis=2) > 0.40

        ak4_jet_cleaning_dr_double = ak.fill_none(abs(ak4_jet_for_cleaning.delta_r(lep_for_cleaning)), True)[:,:,0:2]
        ak4_jet_cleaning_mask_double = ak.min(ak4_jet_cleaning_dr_double, axis=2) > 0.40

        ak4_jets = ak.with_field(ak4_jets, ak4_jet_cleaning_mask_all & ak4_jets.preselected, "cleaned_all")
        ak4_jets = ak.with_field(ak4_jets, ak4_jet_cleaning_mask_single & ak4_jets.preselected, "cleaned_single")
        ak4_jets = ak.with_field(ak4_jets, ak4_jet_cleaning_mask_double & ak4_jets.preselected, "cleaned_double")

        ak4_jets = ak.with_field(ak4_jets, ak4_jets_loose_btag_mask & ak4_jets.cleaned_all, "loose_btag_all")
        ak4_jets = ak.with_field(ak4_jets, ak4_jets_loose_btag_mask & ak4_jets.cleaned_single, "loose_btag_single")
        ak4_jets = ak.with_field(ak4_jets, ak4_jets_loose_btag_mask & ak4_jets.cleaned_double, "loose_btag_double")

        ak4_jets = ak.with_field(ak4_jets, ak4_jets_medium_btag_mask & ak4_jets.cleaned_all, "medium_btag_all")
        ak4_jets = ak.with_field(ak4_jets, ak4_jets_medium_btag_mask & ak4_jets.cleaned_single, "medium_btag_single")
        ak4_jets = ak.with_field(ak4_jets, ak4_jets_medium_btag_mask & ak4_jets.cleaned_double, "medium_btag_double")


        ak4_jets_pre = ak4_jets.mask[ak4_jets.preselected]

        ak4_jets_ptGt50 = ak4_jets_pre.mask[ak4_jets_pre.pt > 50]

        ak8_jets.tau2overtau1 = ak.where(
            (ak.is_none(ak8_jets, axis=1) == 0) & (ak.is_none(ak8_jets.tau2, axis=1) == 0) & (ak.is_none(ak8_jets.tau1, axis=1) == 0),
                ak8_jets.tau2 / ak8_jets.tau1,
                10.0
        )

        ak8_jet_preselection_mask = (
            (ak.is_none(ak8_jets.subjet1) == 0) & (ak.is_none(ak8_jets.subjet2) == 0) &
            (ak8_jets.subjet1.pt >= 20) & (abs(ak8_jets.subjet1.eta) <= 2.4) &
            (ak8_jets.subjet2.pt >= 20) & (abs(ak8_jets.subjet2.eta) <= 2.4) &
            (ak8_jets.jetId > 1) & (ak8_jets.pt >= 200) &
            (abs(ak8_jets.eta) <= 2.4) & (ak8_jets.msoftdrop >= 30) & (ak8_jets.msoftdrop <= 210) &
            (ak8_jets.tau2 / ak8_jets.tau1 <= 0.75)
        )

        ak8_btagDeepB_WP_medium = 0.6321  ### this is 2016
        ak8_jet_btag_mask = (
            (ak8_jets.subjet1.btagDeepB > ak8_btagDeepB_WP_medium) & (ak8_jets.subjet1.pt >= 30)
            | #OR
            (ak8_jets.subjet2.btagDeepB > ak8_btagDeepB_WP_medium) & (ak8_jets.subjet2.pt >= 30)
        )

        ak8_jets = ak.with_field(ak8_jets, ak8_jet_preselection_mask, "preselected")

        leptons_fakeable = ak.concatenate([electrons.mask[electrons.fakeable], muons.mask[muons.fakeable]], axis=1)
        #argsort fails if all values are none, fix by a all none check
        #Check if variables are none, then if all in the nested list are none, then if all in the unnested list are none
        if not ak.all(ak.all(ak.is_none(leptons_fakeable, axis=1), axis=1)):
            leptons_fakeable = leptons_fakeable[ak.argsort(leptons_fakeable.conept, ascending=False)]
        leptons_fakeable = ak.pad_none(leptons_fakeable, 1)

        ak8_jet_lep_pair_for_cleaning = ak.cartesian([ak8_jets.mask[ak8_jets.preselected], leptons_fakeable], nested=True)
        ak8_jet_for_cleaning, lep_for_cleaning = ak.unzip( ak8_jet_lep_pair_for_cleaning )

        ak8_jet_cleaning_dr_all = ak.fill_none(abs(ak8_jet_for_cleaning.delta_r(lep_for_cleaning)), True)
        ak8_jet_cleaning_mask_all = ak.min(ak8_jet_cleaning_dr_all, axis=2) > 0.80

        ak8_jet_cleaning_dr_single = ak8_jet_cleaning_dr_all[:,:,0:1]
        ak8_jet_cleaning_mask_single = ak.min(ak8_jet_cleaning_dr_single, axis=2) > 0.80

        ak8_jet_cleaning_dr_double = ak8_jet_cleaning_dr_all[:,:,0:2]
        ak8_jet_cleaning_mask_double = ak.min(ak8_jet_cleaning_dr_double, axis=2) > 0.80

        ak8_jets = ak.with_field(ak8_jets, ak8_jet_cleaning_mask_all & ak8_jets.preselected, "cleaned_all")
        ak8_jets = ak.with_field(ak8_jets, ak8_jet_cleaning_mask_single & ak8_jets.preselected, "cleaned_single")
        ak8_jets = ak.with_field(ak8_jets, ak8_jet_cleaning_mask_double & ak8_jets.preselected, "cleaned_double")

        ak8_jets = ak.with_field(ak8_jets, ak8_jet_btag_mask & ak8_jets.cleaned_all, "btag_all")
        ak8_jets = ak.with_field(ak8_jets, ak8_jet_btag_mask & ak8_jets.cleaned_single, "btag_single")
        ak8_jets = ak.with_field(ak8_jets, ak8_jet_btag_mask & ak8_jets.cleaned_double, "btag_double")


        events["HT"] = ak.sum(ak4_jets_ptGt50.pt, axis=1)

        ##### Event level selection
        ##### https://github.com/aebid/HHbbWW_Run3/blob/main/python/event_selection.py#L14
        #Prepare leptons
        leptons_preselected = ak.concatenate([electrons.mask[electrons.preselected], muons.mask[muons.preselected]], axis=1)
        leptons_fakeable = ak.concatenate([electrons.mask[electrons.fakeable], muons.mask[muons.fakeable]], axis=1)
        leptons_tight = ak.concatenate([electrons.mask[electrons.tight], muons.mask[muons.tight]], axis=1)

        if ak.any(leptons_preselected): #Required in case no leptons available, can happen with data
            leptons_preselected = leptons_preselected[ak.argsort(leptons_preselected.conept, axis=1, ascending=False)]
        if ak.any(leptons_fakeable):
            leptons_fakeable = leptons_fakeable[ak.argsort(leptons_fakeable.conept, axis=1, ascending=False)]
        if ak.any(leptons_tight):
            leptons_tight = leptons_tight[ak.argsort(leptons_tight.conept, axis=1, ascending=False)]

        leading_leptons = ak.pad_none(leptons_fakeable, 1)[:,0]

        #We break the cuts into separate steps for the cutflow
        #Step 1 -- Require at least 1 fakeable (or tight) lepton
        #Step 2 -- Require MET filters
        #Step 3 -- Leading lepton cone-pT for El (Mu) >= 32.0 (25.0)
        #Step 4 -- Z mass and invariant mass cuts
        #Step 5 -- HLT Cuts
        #   If El, pass El trigger
        #   If Mu, pass Mu trigger
        #Step 6 -- MC match for leading lepton
        #Step 7 -- Require no more than 1 tight lepton (must be leading)
        #Step 8 -- Tau veto
        #Step 9 -- 1 or more btagged ak8_jets or 1 or more btagged ak4_jets
        #Step 10 -- Categories


        events["single_cutflow"] = np.zeros_like(events.run)

        events["is_e"] = (abs(leading_leptons.pdgId) == 11)
        events["is_m"] = (abs(leading_leptons.pdgId) == 13)

        #Require at least 1 fakeable (or tight) lepton
        one_fakeable_lepton = ak.sum(leptons_fakeable.fakeable, axis=1) >= 1
        single_step1_mask = ak.fill_none(one_fakeable_lepton, False)
        listOfCuts.add( "single_lepton", ak.to_numpy(single_step1_mask) )

        single_step2_mask = ak.fill_none(METcleaning, False)

        #Leading lepton cone-pT for El (Mu) >= 32.0 (25.0)
        cone_pt_cuts = ak.where(
            abs(leading_leptons.pdgId) == 11,
                leading_leptons.conept >= 32.0,
                ak.where(
                    abs(leading_leptons.pdgId) == 13,
                        leading_leptons.conept >= 25.0,
                        False
                )
        )
        single_step3_mask = ak.fill_none(cone_pt_cuts, False)
        listOfCuts.add( 'LeadLeptonConePtCut', ak.to_numpy(single_step3_mask) )

        #Z mass and invariant mass cuts
        #No pair of same-flavor, opposite-sign preselcted leptons within 10 GeV of the Z mass (91.1876)
        #Invariant mass of each pair of preselected leptons (electrons not cleaned) must be greater than 12 GeV
        lep_pairs_for_Zmass_and_Invarmass = ak.combinations(leptons_preselected, 2)
        first_leps, second_leps = ak.unzip(lep_pairs_for_Zmass_and_Invarmass)

        lep1_lorentz_vec = ak.zip(
            {
                "pt": ak.fill_none(first_leps.pt, 0.0),
                "eta": ak.fill_none(first_leps.eta, 0.0),
                "phi": ak.fill_none(first_leps.phi, 0.0),
                "mass": ak.fill_none(first_leps.mass, 0.0),
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior,
        )

        lep2_lorentz_vec = ak.zip(
            {
                "pt": ak.fill_none(second_leps.pt, 0.0),
                "eta": ak.fill_none(second_leps.eta, 0.0),
                "phi": ak.fill_none(second_leps.phi, 0.0),
                "mass": ak.fill_none(second_leps.mass, 0.0),
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior,
        )


        Invariant_mass_cut = ak.all(
            (
                ((lep1_lorentz_vec + lep2_lorentz_vec).mass > 12.0) |
                ak.is_none(first_leps, axis = 1) |
                ak.is_none(second_leps, axis = 1)
            ), axis = 1
        )
        Zmass_cut = ak.any(
            (
                (abs(ak.fill_none(first_leps.pdgId, 0)) == abs(ak.fill_none(second_leps.pdgId, 0))) &
                (ak.fill_none(first_leps.charge, 0) != ak.fill_none(second_leps.charge, 0)) &
                (abs((lep1_lorentz_vec + lep2_lorentz_vec).mass - 91.1876) < 10.0)
            ), axis = 1
        ) == 0

        single_step4_mask = ak.fill_none(Invariant_mass_cut & Zmass_cut, False)
        listOfCuts.add( 'ZMassAndInvarMassCut', ak.to_numpy(single_step4_mask) )


        single_step5_mask = ak.fill_none(trigger_selection, False)
        listOfCuts.add( 'PassesHLTCuts', ak.to_numpy(single_step5_mask) )

        #MC match for leading and subleading leptons
        leading_MC_match = leading_leptons.MC_Match | (isMC == False)


        #No more than 1 tight leptons AND should be the same as leading lepton
        n_tight_leptons = ak.sum(leptons_tight.tight, axis=1)

        tight_lep_cut = ((n_tight_leptons == 0) | ((n_tight_leptons == 1) & (leading_leptons.tight)))

        #single_step7_mask = ak.fill_none(n_tight_leptons <= 1, False)
        single_step7_mask = ak.fill_none(tight_lep_cut, False)
        listOfCuts.add( 'AtMostOneTightLep', ak.to_numpy(single_step7_mask) )

        taus = copy.deepcopy(events.Tau)
        #Tau veto: no tau passing pt>20, abs(eta) < 2.3, abs(dxy) <= 1000, abs(dz) <= 0.2, "decayModeFindingNewDMs", decay modes = {0, 1, 2, 10, 11}, and "byMediumDeepTau2017v2VSjet", "byVLooseDeepTau2017v2VSmu", "byVVVLooseDeepTau2017v2VSe". Taus overlapping with fakeable electrons or fakeable muons within dR < 0.3 are not considered for the tau veto
        #False -> Gets Removed : True -> Passes veto
        tau_veto_pairs = ak.cartesian([taus, leptons_fakeable], nested=True)
        taus_for_veto, leps_for_veto = ak.unzip(tau_veto_pairs)

        tau_veto_cleaning = ak.min(abs(taus_for_veto.delta_r(leps_for_veto)), axis=2) >= 0.3

        tau_veto_selection = (
        (taus.pt > 20) & (abs(taus.eta) < 2.3) & (abs(taus.dxy) <= 1000.0) & (abs(taus.dz) <= 0.2) & (getattr(taus, 'idDecayModeNewDMs', False) | getattr(taus, 'idDecayModeOldDMs', False)) &
        (
            (taus.decayMode == 0) | (taus.decayMode == 1) | (taus.decayMode == 2) | (taus.decayMode == 10) | (taus.decayMode == 11)
        ) &
        (taus.idDeepTau2017v2p1VSjet >= 16) & (taus.idDeepTau2017v2p1VSmu >= 1) & (taus.idDeepTau2017v2p1VSe >= 1)
        )

        tau_veto = ak.any(tau_veto_cleaning & tau_veto_selection, axis=1) == 0

        single_step8_mask = ak.fill_none(tau_veto, False)
        listOfCuts.add( 'TauVeto', ak.to_numpy(single_step8_mask) )

        #Jet cuts
        #1 or more btagged ak8_jets or 1 or more btagged ak4_jets
        one_btagged_jet = (ak.sum(ak4_jets.medium_btag_single, axis=1) >= 1) | (ak.sum(ak8_jets.btag_single, axis=1) >= 1)

        single_step9_mask = ak.fill_none(one_btagged_jet, False)
        listOfCuts.add( 'AtLeastOneBJet', ak.to_numpy(single_step9_mask) )

        #Count ak4 jets that are dR 1.2 away from btagged ak8 jets
        #Require either:
        #   0 ak8 btagged jets and 3 or more cleaned ak4 jets
        #   or
        #   1 or more ak8 btagged jets and 1 or more cleaned ak4 jets 1.2dR away from an ak8 bjet
        #ak8_jets_btag_single = ak8_jets.mask[ak8_jets.btag_single]
        ak8_jets_btag_single = ak8_jets[ak8_jets.btag_single]

        ak8_jets_btag_single_sorted = ak8_jets_btag_single
        if ak.any(ak8_jets_btag_single): #Required in case no ak8 jets available, can happen with data
            ak8_jets_btag_single_sorted = ak8_jets_btag_single[ak.argsort(ak8_jets_btag_single.pt, axis=1, ascending=False)]

        ak8_jets_btag_single_sorted_padded = ak.pad_none(ak8_jets_btag_single_sorted, 1)

        ak4_jets_padded = ak.pad_none(ak4_jets, 1)

        clean_ak4_jets_btagged_ak8_jets = ak.cartesian([ak4_jets_padded.mask[ak4_jets_padded.cleaned_single], ak8_jets_btag_single_sorted_padded], nested=True)

        clean_ak4_for_veto, btag_ak8_for_veto = ak.unzip(clean_ak4_jets_btagged_ak8_jets)

        ak4_jets.jets_that_not_bb = ak.any(abs(clean_ak4_for_veto.delta_r(btag_ak8_for_veto)) > 1.2, axis=2)
        n_jets_that_not_bb = ak.sum(ak4_jets.jets_that_not_bb, axis=1)


        jet_btag_veto = (
            ( (n_jets_that_not_bb >= 1) & (ak.sum(ak8_jets.btag_single, axis=1) >= 1) )
            |
            ( (ak.sum(ak8_jets.btag_single, axis=1) == 0) & (ak.sum(ak4_jets.cleaned_single, axis=1) >= 3) )
        )

        single_step10_mask = ak.fill_none(jet_btag_veto, False)
        listOfCuts.add( 'EnoughNonBJets', ak.to_numpy(single_step10_mask) )


        listOfCuts.add( "Single_HbbFat_WjjRes_AllReco", ak.to_numpy( ak.fill_none((ak.sum(ak8_jets.btag_single, axis=1) >= 1) & (n_jets_that_not_bb >= 2), False)) )

        listOfCuts.add( "Single_HbbFat_WjjRes_MissJet", ak.to_numpy( ak.fill_none((ak.sum(ak8_jets.btag_single, axis=1) >= 1) & (n_jets_that_not_bb < 2), False)) )

        listOfCuts.add( "Single_Res_allReco_2b", ak.to_numpy( ak.fill_none((ak.sum(ak8_jets.btag_single, axis=1) == 0) & (ak.sum(ak4_jets.cleaned_single, axis=1) >= 4) & (ak.sum(ak4_jets.medium_btag_single, axis=1) > 1), False)) )

        listOfCuts.add( "Single_Res_allReco_1b", ak.to_numpy( ak.fill_none((ak.sum(ak8_jets.btag_single, axis=1) == 0) & (ak.sum(ak4_jets.cleaned_single, axis=1) >= 4) & (ak.sum(ak4_jets.medium_btag_single, axis=1) == 1), False)) )

        listOfCuts.add( "Single_Res_MissWJet_2b", ak.to_numpy( ak.fill_none((ak.sum(ak8_jets.btag_single, axis=1) == 0) & (ak.sum(ak4_jets.cleaned_single, axis=1) < 4) & (ak.sum(ak4_jets.medium_btag_single, axis=1) > 1), False)) )

        listOfCuts.add( "Single_Res_MissWJet_1b", ak.to_numpy( ak.fill_none((ak.sum(ak8_jets.btag_single, axis=1) == 0) & (ak.sum(ak4_jets.cleaned_single, axis=1) < 4) & (ak.sum(ak4_jets.medium_btag_single, axis=1) == 1), False)) )

        listOfCuts.add( "Single_Signal", ak.to_numpy( ak.fill_none(((ak.sum(leptons_tight.tight, axis=1) == 1) & (leading_leptons.tight)), False)) )

        listOfCuts.add( "Single_Fake", ak.to_numpy( ak.fill_none(((leading_leptons.tight) == 0), False)) )

        ### Gen matching
        genparts = events.GenPart
        genjets = events.GenJet

        # b-quarks from Higgs decay
        bFromH = find_genpart(genparts, [5], [25])

        # H -> WW
        # light quarks W decay
        qFromW = find_genpart(genparts, [1, 2 ,3, 4], [24])

        # leptons from W decay
        lepFromW = find_genpart(genparts, [11, 13, 15], [24, 25])
        nuFromW = find_genpart(genparts, [12, 14, 16], [24, 25])


        ################################################################
        #### End of TAMU code
        ################################################################


        # Weights
        eweight = Weights(len(events),storeIndividual=True)
        if not isData:
            with uproot.open(events.metadata['filename']) as rfile:
                Runs = rfile['Runs']
                genEventSumw = np.sum(Runs['genEventSumw'])

            lumi = events.metadata.get('lumi', 1.0)
            xs   = events.metadata.get('xs', 1.0)
            events['weight'] = events.genWeight * (lumi * xs / genEventSumw)
            eweight.add('genweight', events['genWeight'])

            pu_weight = list(correctionlib.CorrectionSet.from_file(self.corrections_metadata[year]['PU']).values())[0]
            pu = pu_weight.evaluate( events.Pileup.nTrueInt.to_numpy(), 'nominal' )
            eweight.add('pileup',pu)

        def normalize(val, cut):
            if cut is None: return ak.to_numpy(ak.fill_none(val, np.nan))
            else: return ak.to_numpy(ak.fill_none(val[cut], np.nan))

        dataset = events.metadata['dataset']
        self.cutflow[dataset] = {}
        sel_dict = {
            'goldenJson' : listOfCuts.require( goldenJson=True ),
            'PassesMETFilters' : listOfCuts.require( goldenJson=True, PassesMETFilters=True ),
            'single_lepton' : listOfCuts.require( goldenJson=True, PassesMETFilters=True, single_lepton=True ),
            'LeadLeptonConePtCut' : listOfCuts.require( goldenJson=True, PassesMETFilters=True, single_lepton=True, LeadLeptonConePtCut=True ),
            'ZMassAndInvarMassCut' : listOfCuts.require( goldenJson=True, PassesMETFilters=True, single_lepton=True, LeadLeptonConePtCut=True, ZMassAndInvarMassCut=True ),
            'PassesHLTCuts' : listOfCuts.require( goldenJson=True, PassesMETFilters=True, single_lepton=True, LeadLeptonConePtCut=True, ZMassAndInvarMassCut=True, PassesHLTCuts=True ),
            'AtMostOneTightLep' : listOfCuts.require( goldenJson=True, PassesMETFilters=True, single_lepton=True, LeadLeptonConePtCut=True, ZMassAndInvarMassCut=True, PassesHLTCuts=True, AtMostOneTightLep=True ),
            'TauVeto' : listOfCuts.require( goldenJson=True, PassesMETFilters=True, single_lepton=True, LeadLeptonConePtCut=True, ZMassAndInvarMassCut=True, PassesHLTCuts=True, AtMostOneTightLep=True, TauVeto=True ),
            'AtLeastOneBJet' : listOfCuts.require( goldenJson=True, PassesMETFilters=True, single_lepton=True, LeadLeptonConePtCut=True, ZMassAndInvarMassCut=True, PassesHLTCuts=True, AtMostOneTightLep=True, TauVeto=True, AtLeastOneBJet=True ),
            'EnoughNonBJets' : listOfCuts.require( goldenJson=True, PassesMETFilters=True, single_lepton=True, LeadLeptonConePtCut=True, ZMassAndInvarMassCut=True, PassesHLTCuts=True, AtMostOneTightLep=True, TauVeto=True, AtLeastOneBJet=True, EnoughNonBJets=True ),
            'Single_Signal' : listOfCuts.require( goldenJson=True, PassesMETFilters=True, single_lepton=True, LeadLeptonConePtCut=True, ZMassAndInvarMassCut=True, PassesHLTCuts=True, AtMostOneTightLep=True, TauVeto=True, AtLeastOneBJet=True, EnoughNonBJets=True, Single_Signal=True ),
            'Single_Res_allReco_2b' : listOfCuts.require( goldenJson=True, PassesMETFilters=True, single_lepton=True, LeadLeptonConePtCut=True, ZMassAndInvarMassCut=True, PassesHLTCuts=True, AtMostOneTightLep=True, TauVeto=True, AtLeastOneBJet=True, EnoughNonBJets=True, Single_Signal=True, Single_Res_allReco_2b=True ),
        }
        for isel, icut in sel_dict.items():
            iweight = eweight.weight()[icut]
            self.cutflow[dataset][isel] = len(events[icut])

        #### For simplicity, let's redefine some quantities
        icut = sel_dict['Single_Res_allReco_2b']
        iweight = eweight.weight()[icut]
        ak4_alljets = ak4_jets[ak4_jets.cleaned_single]
        ak4_bjets = ak4_jets[ak4_jets.medium_btag_single]
        ak4_nonbjets = ak4_jets[(ak4_jets.cleaned_single) & (~ak4_jets.medium_btag_single)]
        #puId_cuts = ((ak4_nonbjets.puId_dR2Mean < 0.0425) & (ak4_nonbjets.puId_frac01 < 0.99) & (ak4_nonbjets.puId_majW < 0) & (ak4_nonbjets.puId_jetR < 0.8) &(ak4_nonbjets.puId_jetRchg < 0.8) & (ak4_nonbjets.puId_ptD < 0.8) &  (ak4_nonbjets.chEmEF < 0.8)& (ak4_nonbjets.muEF < 0.1) & (ak4_nonbjets.nConstituents > 4) & (ak4_nonbjets.neEmEF < 0.75) & (ak4_nonbjets.neHEF < 0.6) )
        ak4_W_jets = ak.pad_none(ak4_nonbjets,target = 2, axis = 1)
        
        '''ak4_pairs = ak.combinations(ak4_W_jets,2,axis =1)
        ak4_pairs_index = ak.argcombinations(ak4_W_jets,2,axis =1)
        ak4_pairs_sum = ak.Array((ak4_pairs["0"] + ak4_pairs["1"]).mass)
        ak4_pairs_matched = ak4_pairs_index[ak.argsort(abs(80.377-ak4_pairs_sum),ascending = True)]
        ak4_matched1 = ak4_W_jets[ak4_pairs_matched["0"]]
        ak4_matched2 = ak4_W_jets[ak4_pairs_matched["1"]]
        print((ak4_matched1+ak4_matched2).mass)'''
        ## gen matching
        gen_qFromW = ak.pad_none( qFromW, 2 )
        matched_mask1 = (gen_qFromW[:,0].delta_r(ak4_W_jets)< 0.2)
        matched_mask2 = (gen_qFromW[:,1].delta_r(ak4_W_jets)< 0.2)
        matched_mask3 = (gen_qFromW[:,0].delta_r(genjets)< 0.2)
        matched_mask4 = (gen_qFromW[:,1].delta_r(genjets)< 0.2)
        genjets_qFromW1 =  ak.mask(genjets, matched_mask3)
        genjets_qFromW2 =  ak.mask(genjets, matched_mask4)
        jet_matched_q1 = ak.mask(ak4_W_jets,matched_mask1)
        jet_matched_q2 = ak.mask(ak4_W_jets,matched_mask2)
        jet_unmatched = ak.mask(ak4_W_jets,(~matched_mask1 & ~matched_mask2))
        genjet_matched = ak4_W_jets[(0<=ak4_W_jets.genJetIdx)]
        genjet_unmatched = ak4_W_jets[~(0<=ak4_W_jets.genJetIdx)]
        print(genjet_unmatched.genJetIdx)
        #jet_matched_q1W = ak4_W_jets[ (gen_qFromW[:,0].delta_r( ak4_W_jets ) < 0.2 ) ]
        #jet_matched_q2W =ak4_W_jets[ (gen_qFromW[:,1].delta_r( ak4_W_jets ) < 0.2 ) ]
        #jet_matched_q1 = jet_matched_q1W[~ak.is_none(jet_matched_q1W)]
        #jet_matched_q2 = jet_matched_q2W[~ak.is_none(jet_matched_q2W)]                    
      
       # jet1_matched_q1 = ak.sum(jet_matched_q1[:,0])
       # jet2_matched_q1 = ak.sum(jet_matched_q1[:,1])
       # jet3_matched_q1 = ak.sum(jet_matched_q1[:,2])
       # jet4_matched_q1 = ak.sum(jet_matched_q1[:,3])
       # jet5_matched_q1 = ak.sum(jet_matched_q1[:,4])
       # jet6_matched_q1 = ak.sum(jet_matched_q1[:,5])
       # jet1_matched_q2 = ak.sum(jet_matched_q2[:,0])
       # jet2_matched_q2 = ak.sum(jet_matched_q2[:,1])
       # jet3_matched_q2 = ak.sum(jet_matched_q2[:,2])
       # jet4_matched_q2 = ak.sum(jet_matched_q2[:,3])
       # jet5_matched_q2 = ak.sum(jet_matched_q2[:,4])
       # jet6_matched_q2 = ak.sum(jet_matched_q2[:,5])



        self.hists['numJets'].fill( dataset=dataset,
                                   selection=isel,
                                   numJets=normalize(ak.num(ak4_alljets, axis=1), icut),
                                   numBjets=normalize(ak.num(ak4_bjets, axis=1), icut),
                                   numnonBjets=normalize(ak.num(ak4_nonbjets, axis=1), icut),
                                   weight=iweight )
        self.hists['jet1'].fill( dataset=dataset,
                                   selection=isel,
                                   pt=normalize(ak.flatten(genjets_qFromW1[icut].pt,axis=None), None),
                                   eta=normalize(ak.flatten(genjets_qFromW1[icut].eta,axis=None), None),
                                   mass= normalize(ak.flatten(genjets_qFromW1[icut].phi,axis=None),None))
        self.hists['jet1x'].fill(phi = normalize(ak.flatten(jet_matched_q1[icut].phi,axis=None),None),
                                   neHEF = normalize(ak.flatten(jet_matched_q1[icut].neHEF,axis=None),None))
        self.hists['jet1pileup1'].fill(puId_beta = normalize(ak.flatten(genjet_matched[icut].puId_beta,axis=None),None),
                                   puId_dR2 = normalize(ak.flatten(genjet_matched[icut].puId_dR2Mean,axis=None),None),
                                   puId_frac01 = normalize(ak.flatten(genjet_matched[icut].puId_frac01,axis=None),None))
        self.hists['jet1pileup2'].fill(puId_frac02 = normalize(ak.flatten(genjet_matched[icut].puId_frac02,axis=None),None),
                                   puId_frac03 = normalize(ak.flatten(genjet_matched[icut].puId_frac03,axis=None),None),
                                   puId_frac04 = normalize(ak.flatten(genjet_matched[icut].puId_frac04,axis=None),None))
        self.hists['jet1pileup3'].fill(puId_jetR = normalize(ak.flatten(genjet_matched[icut].puId_jetR,axis=None),None),
                                   puId_jetRchg = normalize(ak.flatten(genjet_matched[icut].puId_jetRchg,axis=None),None),
                                   puId_majW = normalize(ak.flatten(genjet_matched[icut].puId_majW,axis=None),None))
        self.hists['jet1pileup4'].fill(puId_minW = normalize(ak.flatten(genjet_matched[icut].puId_minW,axis=None),None),
                                   puId_nCharged = normalize(ak.flatten(genjet_matched[icut].puId_nCharged,axis=None),None),
                                   puId_ptD = normalize(ak.flatten(genjet_matched[icut].puId_ptD,axis=None),None))
        self.hists['jet1pileup5'].fill(puId_pull = normalize(ak.flatten(genjet_matched[icut].puId_pull,axis=None),None),
                                   chEmEF = normalize(ak.flatten(genjet_matched[icut].chEmEF,axis=None),None),
                                   chHEF = normalize(ak.flatten(genjet_matched[icut].chHEF,axis=None),None))
        self.hists['jet1pileup6'].fill(muEF = normalize(ak.flatten(genjet_matched[icut].muEF,axis=None),None),
                                   nConstituents = normalize(ak.flatten(genjet_matched[icut].nConstituents,axis=None),None),
                                   neEmEF = normalize(ak.flatten(genjet_matched[icut].neEmEF,axis=None),None))

        self.hists['jet1Ids'].fill( jetId = normalize(ak.flatten(genjet_matched[icut].jetId, axis=None),None),
                                    puId= normalize(ak.flatten(genjet_matched[icut].puIdDisc, axis=None), None))
        self.hists['jet1_'].fill( dataset=dataset,
                                   selection=isel,
                                   pt=normalize(ak.flatten(genjets_qFromW2[icut].pt,axis=None), None),
                                   eta=normalize(ak.flatten(genjets_qFromW2[icut].eta,axis=None), None),
                                   mass= normalize(ak.flatten(genjets_qFromW2[icut].phi,axis=None),None))
        self.hists['jet1_x'].fill(phi= normalize(ak.flatten(jet_matched_q2[icut].phi,axis=None),None),
                                   neHEF = normalize(ak.flatten(jet_matched_q2[icut].neHEF,axis=None),None))
        self.hists['jet1_pileup1'].fill(puId_beta = normalize(ak.flatten(genjet_unmatched[icut].puId_beta,axis=None),None),
                                   puId_dR2 = normalize(ak.flatten(genjet_unmatched[icut].puId_dR2Mean,axis=None),None),
                                   puId_frac01 = normalize(ak.flatten(genjet_unmatched[icut].puId_frac01,axis=None),None))
        self.hists['jet1_pileup2'].fill(puId_frac02 = normalize(ak.flatten(genjet_unmatched[icut].puId_frac02,axis=None),None),
                                   puId_frac03 = normalize(ak.flatten(genjet_unmatched[icut].puId_frac03,axis=None),None),
                                   puId_frac04 = normalize(ak.flatten(genjet_unmatched[icut].puId_frac04,axis=None),None))
        self.hists['jet1_pileup3'].fill(puId_jetR = normalize(ak.flatten(genjet_unmatched[icut].puId_jetR,axis=None),None),
                                   puId_jetRchg = normalize(ak.flatten(genjet_unmatched[icut].puId_jetRchg,axis=None),None),
                                   puId_majW = normalize(ak.flatten(genjet_unmatched[icut].puId_majW,axis=None),None))
        self.hists['jet1_pileup4'].fill(puId_minW = normalize(ak.flatten(genjet_unmatched[icut].puId_minW,axis=None),None),
                                   puId_nCharged = normalize(ak.flatten(genjet_unmatched[icut].puId_nCharged,axis=None),None),
                                   puId_ptD = normalize(ak.flatten(genjet_unmatched[icut].puId_ptD,axis=None),None))
        self.hists['jet1_pileup5'].fill(puId_pull = normalize(ak.flatten(genjet_unmatched[icut].puId_pull,axis=None),None),
                                   chEmEF = normalize(ak.flatten(genjet_unmatched[icut].chEmEF,axis=None),None),
                                   chHEF = normalize(ak.flatten(genjet_unmatched[icut].chHEF,axis=None),None))
        self.hists['jet1_pileup6'].fill(muEF = normalize(ak.flatten(genjet_unmatched[icut].muEF,axis=None),None),
                                   nConstituents = normalize(ak.flatten(genjet_unmatched[icut].nConstituents,axis=None),None),
                                   neEmEF = normalize(ak.flatten(genjet_unmatched[icut].neEmEF,axis=None),None))

        self.hists['jet1_Ids'].fill(jetId = normalize(ak.flatten(genjet_unmatched[icut].jetId, axis=None), None),
                                    puId= normalize(ak.flatten(genjet_unmatched[icut].puIdDisc, axis=None), None))
        '''self.hists['jet1W'].fill( pt=normalize(ak4_matched1[icut][:,0].matched_gen.pt,None),
                                   eta=normalize(ak4_matched1[icut][:,0].eta,None),
                                   mass= normalize(ak4_matched1[icut][:,0].phi,None))'''
        '''self.hists['jet2W'].fill( pt=normalize(ak4_matched2[icut][:,0].matched_gen.pt,None),
                                   eta=normalize(ak4_matched2[icut][:,0].eta,None),
                                   mass= normalize(ak4_matched2[icut][:,0].phi,None))'''
        self.hists['jet2Wx'].fill(phi= normalize(ak.flatten(ak4_W_jets[icut].phi,axis=None),None))
        self.hists['jet2WIds'].fill(dataset=dataset,
                                    jetId = normalize(ak.flatten(genjet_matched[icut].jetId,axis=None), None),
                                    puId= normalize(ak.flatten(genjet_matched[icut].puIdDisc,axis=None), None))
        self.hists['jet3W'].fill(  pt=normalize(ak4_nonbjets[icut][:,0].matched_gen.pt,None),
                                   eta=normalize(ak4_nonbjets[icut][:,0].eta,None),
                                   mass= normalize(ak4_nonbjets[icut][:,0].phi,None))
        # self.hists['jet3Wx'].fill(phi= normalize(ak.flatten(jet_matched_q2[icut].phi,axis=None),None))
                                   
        self.hists['jet3WIds'].fill( dataset=dataset,
                                     jetId = normalize(ak.flatten(genjet_unmatched[icut].jetId,axis=None), None),
                                    puId= normalize(ak.flatten(genjet_unmatched[icut].puIdDisc,axis=None), None))
        '''self.hists['jet4W'].fill( pt=normalize(jet_matched_q2[icut][:,1].matched_gen.pt,None),
                                   eta=normalize(jet_matched_q2[icut][:,1].eta,None),
                                   mass= normalize(jet_matched_q2[icut][:,1].phi,None))'''
       # self.hists['jet4WIds'].fill( dataset=dataset,
        #                            jetId = normalize(ak4_W_jets[icut][:,3].jetId, None),
         #                           puId= normalize(ak4_W_jets[icut][:,3].puId, None),
          #                          qgl = normalize(ak4_W_jets[icut][:,3].qgl,None),
           #                         weight=iweight )



                                     
         
        output = {
            'hists' : self.hists,
            'cutflow': self.cutflow,
            'nEvent' : {
             dataset : len(events)
            }
        }

        return output
    

    def postprocess(self, accumulator):
        return accumulator


#### from https://github.com/aebid/HHbbWW_Run3/blob/main/python/genparticles.py#L42
def find_genpart(genpart, pdgid, ancestors):
    """
    Find gen level particles given pdgId (and ancestors ids)

    Parameters:
    genpart (GenPart): NanoAOD GenPart collection.
    pdgid (list): pdgIds for the target particles.
    idmother (list): pdgIds for the ancestors of the target particles.

    Returns:
    NanoAOD GenPart collection
    """

    def check_id(p):
        return np.abs(genpart.pdgId) == p

    pid = reduce(np.logical_or, map(check_id, pdgid))

    if ancestors:
        ancs, ancs_idx = [], []
        for i, mother_id in enumerate(ancestors):
            if i == 0:
                mother_idx = genpart[pid].genPartIdxMother
            else:
                mother_idx = genpart[ancs_idx[i-1]].genPartIdxMother
            ancs.append(np.abs(genpart[mother_idx].pdgId) == mother_id)
            ancs_idx.append(mother_idx)

        decaymatch =  reduce(np.logical_and, ancs)
        return genpart[pid][decaymatch]

    return genpart[pid]


### extra functions
def mask_event_decision(event, decision='OR', branch='HLT', list_to_mask=[''], list_to_skip=['']):
    '''
    Takes event.branch and passes an boolean array mask with the decisions of all the list_to_mask
    '''

    tmp_list = []
    if branch in event.fields:
        for i in list_to_mask:
            if i in event[branch].fields:
                tmp_list.append( event[branch][i] )
            elif i in list_to_skip: continue
            else: logging.warning(f'\n{i} branch not in {branch} for event.')
    else: logging.warning(f'\n{branch} branch not in event.')
    tmp_array = np.array( tmp_list )

    if decision.lower().startswith('or'): decision_array = np.any( tmp_array, axis=0 )
    else: decision_array = np.all( tmp_array, axis=0 )

    return decision_array
