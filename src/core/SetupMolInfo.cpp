/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2012-2019 The plumed team
   (see the PEOPLE file at the root of the distribution for a list of names)

   See http://www.plumed.org for more information.

   This file is part of plumed, version 2.

   plumed is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   plumed is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with plumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
#include "SetupMolInfo.h"
#include "Atoms.h"
#include "ActionRegister.h"
#include "ActionSet.h"
#include "PlumedMain.h"
#include "tools/MolDataClass.h"
#include "tools/PDB.h"


namespace PLMD {


/*
This action is defined in core/ as it is used by other actions.
Anyway, it is registered in setup/, so that excluding that module from
compilation will exclude it from plumed.
*/


void SetupMolInfo::registerKeywords( Keywords& keys ) {
  ActionSetup::registerKeywords(keys);
  keys.add("compulsory","STRUCTURE","a file in pdb format containing a reference structure. "
           "This is used to defines the atoms in the various residues, chains, etc . "
           "For more details on the PDB file format visit http://www.wwpdb.org/docs.html");
  keys.add("compulsory","MOLTYPE","protein","what kind of molecule is contained in the pdb file - usually not needed since protein/RNA/DNA are compatible");
  keys.add("atoms","CHAIN","(for masochists ( mostly Davide Branduardi ) ) The atoms involved in each of the chains of interest in the structure.");
}

SetupMolInfo::~SetupMolInfo() {
// empty destructor to delete unique_ptr
}

SetupMolInfo::SetupMolInfo( const ActionOptions&ao ):
  Action(ao),
  ActionSetup(ao),
  ActionAtomistic(ao)
{
  // Read what is contained in the pdb file
  parse("MOLTYPE",mytype);

  std::vector<SetupMolInfo*> moldat=plumed.getActionSet().select<SetupMolInfo*>();
  if( moldat.size()!=0 ) error("cannot use more than one MOLINFO action in input");

  std::vector<AtomNumber> backbone;
  parseAtomList("CHAIN",backbone);
  if( read_backbone.size()==0 ) {
    for(unsigned i=1;; ++i) {
      parseAtomList("CHAIN",i,backbone);
      if( backbone.size()==0 ) break;
      read_backbone.push_back(backbone);
      backbone.resize(0);
    }
  } else {
    read_backbone.push_back(backbone);
  }
  if( read_backbone.size()==0 ) {
    std::string reference; parse("STRUCTURE",reference);

    if( ! pdb.read(reference,plumed.getAtoms().usingNaturalUnits(),0.1/plumed.getAtoms().getUnits().getLength()))plumed_merror("missing input file " + reference );

    std::vector<std::string> chains; pdb.getChainNames( chains );
    log.printf("  pdb file named %s contains %u chains \n",reference.c_str(), static_cast<unsigned>(chains.size()));
    for(unsigned i=0; i<chains.size(); ++i) {
      unsigned start,end; std::string errmsg;
      pdb.getResidueRange( chains[i], start, end, errmsg );
      if( errmsg.length()!=0 ) error( errmsg );
      AtomNumber astart,aend;
      pdb.getAtomRange( chains[i], astart, aend, errmsg );
      if( errmsg.length()!=0 ) error( errmsg );
      log.printf("  chain named %s contains residues %u to %u and atoms %u to %u \n",chains[i].c_str(),start,end,astart.serial(),aend.serial());
    }
  }
}

void SetupMolInfo::getBackbone( std::vector<std::string>& restrings, const std::string& fortype, std::vector< std::vector<AtomNumber> >& backbone ) {
  if( fortype!=mytype ) error("cannot calculate a variable designed for " + fortype + " molecules for molecule type " + mytype );
  if( MolDataClass::numberOfAtomsPerResidueInBackbone( mytype )==0 ) error("backbone is not defined for molecule type " + mytype );

  if( read_backbone.size()!=0 ) {
    if( restrings.size()!=1 ) error("cannot interpret anything other than all for residues when using CHAIN keywords");
    if( restrings[0]!="all" ) error("cannot interpret anything other than all for residues when using CHAIN keywords");
    backbone.resize( read_backbone.size() );
    for(unsigned i=0; i<read_backbone.size(); ++i) {
      backbone[i].resize( read_backbone[i].size() );
      for(unsigned j=0; j<read_backbone[i].size(); ++j) backbone[i][j]=read_backbone[i][j];
    }
  } else {
    bool useter=false; // This is used to deal with terminal groups in WHOLEMOLECULES
    if( restrings.size()==1 ) {
      useter=( restrings[0].find("ter")!=std::string::npos );
      if( restrings[0].find("all")!=std::string::npos ) {
        std::vector<std::string> chains; pdb.getChainNames( chains );
        for(unsigned i=0; i<chains.size(); ++i) {
          unsigned r_start, r_end; std::string errmsg, mm, nn;
          pdb.getResidueRange( chains[i], r_start, r_end, errmsg );
          if( !useter ) {
            std::string resname = pdb.getResidueName( r_start );
            if( MolDataClass::isTerminalGroup( mytype, resname ) ) r_start++;
            resname = pdb.getResidueName( r_end );
            if( MolDataClass::isTerminalGroup( mytype, resname ) ) r_end--;
          }
          Tools::convert(r_start,mm); Tools::convert(r_end,nn);
          if(i==0) restrings[0] = mm + "-" + nn;
          else restrings.push_back(  mm + "-" + nn );
        }
      }
    }
    Tools::interpretRanges(restrings);

    // Convert the list of involved residues into a list of segments of chains
    int nk, nj; std::vector< std::vector<unsigned> > segments;
    std::vector<unsigned> thissegment;
    Tools::convert(restrings[0],nk); thissegment.push_back(nk);
    for(unsigned i=1; i<restrings.size(); ++i) {
      Tools::convert(restrings[i-1],nk);
      Tools::convert(restrings[i],nj);
      if( (nk+1)!=nj || pdb.getChainID(nk)!=pdb.getChainID(nj) ) {
        segments.push_back(thissegment);
        thissegment.resize(0);
      }
      thissegment.push_back(nj);
    }
    segments.push_back( thissegment );

    // And now get the backbone atoms from each segment
    backbone.resize( segments.size() );
    std::vector<AtomNumber> atomnumbers;
    for(unsigned i=0; i<segments.size(); ++i) {
      for(unsigned j=0; j<segments[i].size(); ++j) {
        std::string resname=pdb.getResidueName( segments[i][j] );
        if( !MolDataClass::allowedResidue(mytype, resname) ) {
          std::string num; Tools::convert( segments[i][j], num );
          error("residue " + num + " is not recognized for moltype " + mytype );
        }
        if( !useter && MolDataClass::isTerminalGroup( mytype, resname ) ) {
          std::string num; Tools::convert( segments[i][j], num );
          error("residue " + num + " appears to be a terminal group");
        }
        if( resname=="GLY" ) warning("GLY residues are achiral - assuming HA1 atom is in CB position");
        MolDataClass::getBackboneForResidue( mytype, segments[i][j], pdb, atomnumbers );
        if( atomnumbers.size()==0 ) {
          std::string num; Tools::convert( segments[i][j], num );
          error("Could not find required backbone atom in residue number " + num );
        } else {
          for(unsigned k=0; k<atomnumbers.size(); ++k) backbone[i].push_back( atomnumbers[k] );
        }
        atomnumbers.resize(0);
      }
    }
  }
}

void SetupMolInfo::interpretSymbol( const std::string& symbol, std::vector<AtomNumber>& atoms )const {
  MolDataClass::specialSymbol( mytype, symbol, pdb, atoms );
}

std::string SetupMolInfo::getAtomName(AtomNumber a)const {
  return pdb.getAtomName(a);
}

unsigned SetupMolInfo::getResidueNumber(AtomNumber a)const {
  return pdb.getResidueNumber(a);
}

std::string SetupMolInfo::getResidueName(AtomNumber a)const {
  return pdb.getResidueName(a);
}

}
