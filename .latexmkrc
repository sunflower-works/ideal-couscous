add_cus_dep('glo','gls',0,'makeglossaries');
sub makeglossaries {
  my ($base) = @_;
  system("makeglossaries $base");
}

