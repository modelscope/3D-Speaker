#!/usr/bin/perl
#
# Copyright 2018  Ewald Enzinger
#           2022  Hongji Wang
#
# Usage: m4a2wav.pl /export/voxceleb2_m4a dev /export/voxceleb2_wav
#
# Note: This script requires ffmpeg to be installed and its location included in $PATH.

if (@ARGV != 3) {
  print STDERR "Usage: $0 <path-to-voxceleb2_m4a> <dataset> <path-to-voxceleb2_wav>\n";
  print STDERR "e.g. $0 /export/voxceleb2_m4a dev /export/voxceleb2_wav\n";
  exit(1);
}

# Check that ffmpeg is installed.
if (`which ffmpeg` eq "") {
  die "Error: this script requires that ffmpeg is installed.";
}

($database_m4a, $dataset, $database_wav) = @ARGV;

if ("$dataset" ne "dev" && "$dataset" ne "test") {
  die "dataset parameter must be 'dev' or 'test'!";
}

opendir my $dh, "$database_m4a/$dataset/aac" or die "Cannot open directory: $!";
my @spkr_dirs = grep {-d "$database_m4a/$dataset/aac/$_" && ! /^\.{1,2}$/} readdir($dh);
closedir $dh;

if (system("mkdir -p $database_wav/$dataset") != 0) {
  die "Error making directory $database_wav/$dataset";
}

open(fp, ">", "$database_wav/$dataset/m4a2wav_$dataset.sh") or die "Could not open the output file $database_wav/m4a2wav.sh";

foreach (@spkr_dirs) {
  my $spkr_id = $_;

  opendir my $dh, "$database_m4a/$dataset/aac/$spkr_id/" or die "Cannot open directory: $!";
  my @rec_dirs = grep {-d "$database_m4a/$dataset/aac/$spkr_id/$_" && ! /^\.{1,2}$/} readdir($dh);
  closedir $dh;

  foreach (@rec_dirs) {
    my $rec_id = $_;

    opendir my $dh, "$database_m4a/$dataset/aac/$spkr_id/$rec_id/" or die "Cannot open directory: $!";
    my @files = map{s/\.[^.]+$//;$_}grep {/\.m4a$/} readdir($dh);
    closedir $dh;

    foreach (@files) {
      my $name = $_;
      if ( not -e "$database_wav/$dataset/aac/$spkr_id/$rec_id"){
          system("mkdir -p $database_wav/$dataset/aac/$spkr_id/$rec_id");
      }
      my $wav = "ffmpeg -v 8 -i $database_m4a/$dataset/aac/$spkr_id/$rec_id/$name.m4a -f wav -acodec pcm_s16le $database_wav/$dataset/aac/$spkr_id/$rec_id/$name.wav";
      print fp "$wav", "\n";
    }
  }
}
close(fp) or die;

# generate wav
#system("sh $database_wav/$dataset/m4a2wav_$dataset.sh");
